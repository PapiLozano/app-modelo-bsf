import 'dart:math' as math;
import 'dart:typed_data';
import 'dart:ui' show Offset;

import 'package:flutter/foundation.dart' show debugPrint, compute;
import 'package:flutter_onnxruntime/flutter_onnxruntime.dart';
import 'package:image/image.dart' as img;

import 'coin_detector.dart';
import 'models.dart';
import 'package:opencv_core/opencv.dart' as cv;

// id 0 = placeholder de Roboflow, nunca se usa.
const Map<int, String> _classNames = {
  1: 'imago',
  2: 'larva',
  3: 'prepupa',
  4: 'pupa',
};

const double _scoreThreshold = 0.40;
const double _nmsIouThreshold = 0.5;

/// Porta a Dart la misma lógica de decodificación que ya validamos en
/// Python (comparar_onnx.py): sigmoid sobre labels/dets, conversión
/// cxcywh -> xyxy, y NMS por clase. Además decodifica la salida de
/// máscaras de segmentación y extrae:
///   - su CONTORNO (simplificado con approxPolyDP + suavizado con
///     Chaikin) para dibujarla bien y rápido,
///   - su LONGITUD real siguiendo la curva del cuerpo (esqueleto vía
///     Zhang-Suen, con reserva a `minAreaRect` si `ximgproc` no está
///     disponible en el build).
///
/// NOTA DE RENDIMIENTO: todo el trabajo de `_decodeHeavyWork` (NMS +
/// máscaras + esqueleto, la parte realmente pesada) corre en un
/// `Isolate` en segundo plano vía `compute()`, para no congelar la UI.
/// La inferencia ONNX en sí (`session.run`) se queda en el isolate
/// principal porque el plugin `flutter_onnxruntime` no está garantizado
/// isolate-safe fuera de él.
class RfDetrService {
  static const int inputSize = 504;

  static const List<double> imagenetMean = [0.485, 0.456, 0.406];
  static const List<double> imagenetStd = [0.229, 0.224, 0.225];

  final OnnxRuntime _ort = OnnxRuntime();
  OrtSession? _session;

  Future<void> loadModel() async {
    // Intentamos usar el acelerador de hardware nativo de cada
    // plataforma (NNAPI en Android, CoreML en iOS) y caemos a CPU si
    // el dispositivo no lo soporta o el provider falla al inicializar.
    //
    // OJO: no pude verificar en documentación los nombres EXACTOS de
    // los valores del enum OrtProvider más allá de `CPU` (confirmado
    // en la guía oficial). Si `NNAPI`/`CoreML` no compilan tal cual,
    // abre el autocompletado de tu IDE sobre `OrtProvider.` y usa los
    // nombres reales que aparezcan (case-sensitive) — es la única
    // parte de este cambio que no pude confirmar al 100%.
    try {
      final options = OrtSessionOptions(
        providers: [OrtProvider.NNAPI, OrtProvider.CORE_ML, OrtProvider.CPU],
      );
      _session = await _ort.createSessionFromAsset(
        'assets/models/inference_model.onnx',
        options: options,
      );
      debugPrint('Sesión ONNX creada con providers NNAPI/CoreML/CPU.');
    } catch (e) {
      debugPrint(
        'No se pudo crear la sesión con providers NNAPI/CoreML ($e). '
        'Cayendo a la configuración por defecto (CPU).',
      );
      _session = await _ort.createSessionFromAsset(
        'assets/models/inference_model.onnx',
      );
    }
  }

  bool get isReady => _session != null;

  Future<AnalysisOutput> analyze(img.Image original) async {
    final session = _session;
    if (session == null) {
      throw StateError('Modelo no cargado. Llama a loadModel() primero.');
    }

    final resized = img.copyResize(
      original,
      width: inputSize,
      height: inputSize,
    );

    final inputData = Float32List(1 * 3 * inputSize * inputSize);
    int idx = 0;
    for (int c = 0; c < 3; c++) {
      for (int y = 0; y < inputSize; y++) {
        for (int x = 0; x < inputSize; x++) {
          final pixel = resized.getPixel(x, y);
          double v = c == 0
              ? pixel.r / 255.0
              : c == 1
              ? pixel.g / 255.0
              : pixel.b / 255.0;
          v = (v - imagenetMean[c]) / imagenetStd[c];
          inputData[idx++] = v;
        }
      }
    }

    final inputTensor = await OrtValue.fromList(inputData, [
      1,
      3,
      inputSize,
      inputSize,
    ]);

    final inputName = session.inputNames.first;
    final outputs = await session.run({inputName: inputTensor});

    debugPrint('Salidas del modelo: ${outputs.keys.toList()}');

    final detsOut = await outputs['dets']!.asList();
    final labelsOut = await outputs['labels']!.asList();

    final masksValue = _findOutput(outputs, [
      'masks',
      'pred_masks',
      'mask',
      'segm_masks',
    ]);
    List<dynamic>? masksBatch;
    if (masksValue != null) {
      final masksOut = await masksValue.asList();
      masksBatch = masksOut[0] as List;
      if (masksBatch.isNotEmpty) {
        final firstMask = masksBatch[0] as List;
        final hm = firstMask.length;
        final wm = hm > 0 ? (firstMask[0] as List).length : 0;
        debugPrint('Resolución de máscara por query: ${hm}x$wm');
      }
    } else {
      debugPrint(
        'No se encontró una salida de máscaras reconocible en el modelo '
        '(se buscaron los nombres: masks, pred_masks, mask, segm_masks).',
      );
    }

    await inputTensor.dispose();
    for (final v in outputs.values) {
      await v.dispose();
    }

    final detsBatch = detsOut[0] as List;
    final labelsBatch = labelsOut[0] as List;

    // ---- Aquí es donde se corta el trabajo hacia el Isolate ----
    // dets/labels/masks a esta altura son solo listas anidadas de
    // números (nada de Mat/OrtValue nativos), así que son 100%
    // transferibles a un isolate en segundo plano sin problema.
    final decoded = await compute(
      _decodeHeavyWork,
      _DecodeTaskInput(
        dets: detsBatch,
        labels: labelsBatch,
        masks: masksBatch,
        origW: original.width,
        origH: original.height,
      ),
    );

    CoinCalibration? calibration;
    try {
      final coin = CoinDetector.findCoin(original);
      if (coin != null) {
        calibration = CoinCalibration(
          centerX: coin.cx,
          centerY: coin.cy,
          radiusPx: coin.r,
          coinDiameterMm: CoinDetector.defaultCoinDiameterMm,
        );
        applyCalibration(decoded.detections, calibration);
      } else {
        debugPrint('No se detectó ninguna moneda de referencia en la foto.');
      }
    } catch (e) {
      debugPrint('Error al calibrar con la moneda: $e');
    }

    return AnalysisOutput(
      detections: decoded.detections,
      counts: decoded.counts,
      calibration: calibration,
    );
  }

  OrtValue? _findOutput(
    Map<String, OrtValue> outputs,
    List<String> candidates,
  ) {
    for (final key in candidates) {
      if (outputs.containsKey(key)) return outputs[key];
    }
    return null;
  }

  static void applyCalibration(
    List<DetectedObject> detections,
    CoinCalibration calibration,
  ) {
    for (final d in detections) {
      final areaPx =
          _polygonArea(
            d.maskContours.isNotEmpty ? d.maskContours.first : null,
          ) ??
          (d.widthPx * d.heightPx);
      final equivDiameterPx = 2 * math.sqrt(areaPx / math.pi);
      d.sizeMm = equivDiameterPx / calibration.pxPerMm;

      if (d.lengthPx != null) {
        d.lengthMm = d.lengthPx! / calibration.pxPerMm;
      }
    }
  }

  static double? _polygonArea(List<Offset>? poly) {
    if (poly == null || poly.length < 3) return null;
    double sum = 0;
    for (int i = 0; i < poly.length; i++) {
      final p1 = poly[i];
      final p2 = poly[(i + 1) % poly.length];
      sum += p1.dx * p2.dy - p2.dx * p1.dy;
    }
    return sum.abs() / 2;
  }
}

// =============================================================
// A PARTIR DE AQUÍ: todo top-level (o estático puro), sin acceso a
// `this` ni a ningún estado de instancia. Es requisito de `compute()`
// -> Flutter necesita poder mandar esta función a un Isolate nuevo.
// =============================================================

class _DecodeTaskInput {
  final List<dynamic> dets;
  final List<dynamic> labels;
  final List<dynamic>? masks;
  final int origW;
  final int origH;

  _DecodeTaskInput({
    required this.dets,
    required this.labels,
    required this.masks,
    required this.origW,
    required this.origH,
  });
}

/// Punto de entrada para `compute()`. Reproduce EXACTAMENTE lo que
/// antes hacía `RfDetrService._decode` de forma síncrona en el isolate
/// principal, pero ahora corre en un isolate aparte.
AnalysisOutput _decodeHeavyWork(_DecodeTaskInput input) {
  const numQueries = 200;
  const numLabelSlots = 5;

  final candidates = <_RawDetection>[];

  for (int q = 0; q < numQueries; q++) {
    final labelRow = input.labels[q] as List;
    final detRow = input.dets[q] as List;

    double bestScore = -1;
    int bestClass = -1;

    for (int c = 1; c < numLabelSlots; c++) {
      final logit = (labelRow[c] as num).toDouble();
      final score = _sigmoid(logit);
      if (score > bestScore) {
        bestScore = score;
        bestClass = c;
      }
    }

    if (bestScore < _scoreThreshold) continue;

    final cx = (detRow[0] as num).toDouble();
    final cy = (detRow[1] as num).toDouble();
    final w = (detRow[2] as num).toDouble();
    final h = (detRow[3] as num).toDouble();

    double x1 = (cx - w / 2) * input.origW;
    double y1 = (cy - h / 2) * input.origH;
    double x2 = (cx + w / 2) * input.origW;
    double y2 = (cy + h / 2) * input.origH;

    x1 = x1.clamp(0, input.origW.toDouble());
    y1 = y1.clamp(0, input.origH.toDouble());
    x2 = x2.clamp(0, input.origW.toDouble());
    y2 = y2.clamp(0, input.origH.toDouble());

    candidates.add(
      _RawDetection(
        classId: bestClass,
        confidence: bestScore,
        x1: x1,
        y1: y1,
        x2: x2,
        y2: y2,
        queryIndex: q,
      ),
    );
  }

  final kept = _nmsPerClass(candidates);

  final detections = kept.map((d) {
    List<List<Offset>> contours = const [];
    double? lengthPx;

    if (input.masks != null && d.queryIndex < input.masks!.length) {
      try {
        final maskQuery = input.masks![d.queryIndex] as List;
        final decodedMask = _decodeMask(maskQuery, input.origW, input.origH);
        contours = decodedMask.contours;
        lengthPx = decodedMask.lengthPx;
      } catch (e) {
        debugPrint(
          'No se pudo decodificar la máscara de la query ${d.queryIndex}: $e',
        );
      }
    }

    final obj = DetectedObject(
      classId: d.classId,
      className: _classNames[d.classId] ?? 'clase_${d.classId}',
      confidence: d.confidence,
      x1: d.x1,
      y1: d.y1,
      x2: d.x2,
      y2: d.y2,
      maskContours: contours,
    );
    obj.lengthPx = lengthPx;
    return obj;
  }).toList();

  final countsMap = <int, int>{};
  for (final d in detections) {
    countsMap[d.classId] = (countsMap[d.classId] ?? 0) + 1;
  }

  final counts = countsMap.entries
      .map(
        (e) => DetectionCount(
          className: _classNames[e.key] ?? 'clase_${e.key}',
          count: e.value,
        ),
      )
      .toList();

  return AnalysisOutput(detections: detections, counts: counts);
}

double _sigmoid(double x) => 1.0 / (1.0 + math.exp(-x));

/// Convierte la máscara cruda de una query (logits [Hm][Wm]) en:
///   - contornos SIMPLIFICADOS (approxPolyDP) y suavizados (Chaikin)
///   - la longitud real del cuerpo (esqueleto) en píxeles
///
/// Sigue el mismo pipeline de Python: sigmoid -> resize bilineal a
/// tamaño completo -> umbral 0.5 -> findContours. A partir de ahí:
///   - `approxPolyDP` reduce el número de vértices del contorno (menos
///     puntos = fillPoly/polylines/hit-testing/Chaikin más rápidos,
///     sin perder la forma real, solo el detalle sub-píxel),
///   - Chaikin redondea lo que queda del efecto "escalera",
///   - `ximgproc.thinning` (Zhang-Suen) da el esqueleto para sumar la
///     distancia real siguiendo la curva del cuerpo. Si `ximgproc` no
///     está disponible en el build, cae de vuelta al lado más largo de
///     `minAreaRect` (aproximación en línea recta).
_MaskDecodeResult _decodeMask(List<dynamic> maskQuery, int origW, int origH) {
  final hm = maskQuery.length;
  final wm = hm > 0 ? (maskQuery[0] as List).length : 0;
  if (hm == 0 || wm == 0) return const _MaskDecodeResult([], null);

  final probFlat = Float32List(hm * wm);
  int idx = 0;
  for (int y = 0; y < hm; y++) {
    final row = maskQuery[y] as List;
    for (int x = 0; x < wm; x++) {
      probFlat[idx++] = _sigmoid((row[x] as num).toDouble());
    }
  }

  final probMat = cv.Mat.fromList(hm, wm, cv.MatType.CV_32FC1, probFlat);

  final resized = cv.resize(probMat, (
    origW,
    origH,
  ), interpolation: cv.INTER_LINEAR);
  probMat.dispose();

  final (_, threshFloat) = cv.threshold(resized, 0.5, 1.0, cv.THRESH_BINARY);
  resized.dispose();

  final mask8u = threshFloat.convertTo(cv.MatType.CV_8UC1, alpha: 255.0);
  threshFloat.dispose();

  final (contours, _) = cv.findContours(
    mask8u,
    cv.RETR_EXTERNAL,
    cv.CHAIN_APPROX_SIMPLE,
  );

  if (contours.isEmpty) {
    mask8u.dispose();
    return const _MaskDecodeResult([], null);
  }

  final rawContours = contours.where((c) => c.length >= 3).map((c) {
    // approxPolyDP: reduce vértices redundantes ANTES de
    // suavizar. epsilon = 0.5% del perímetro -> lo bastante
    // chico para no deformar la silueta, lo bastante grande
    // para tirar los puntos casi-colineales del "escalón".
    final perimeter = cv.arcLength(c, true);
    final epsilon = math.max(0.5, perimeter * 0.005);
    final simplified = cv.approxPolyDP(c, epsilon, true);
    return simplified
        .map((p) => Offset(p.x.toDouble(), p.y.toDouble()))
        .toList();
  }).toList()..sort((a, b) => _shoelaceArea(b).compareTo(_shoelaceArea(a)));

  // Suavizado de Chaikin sobre el contorno YA simplificado (mucho
  // menos puntos que antes -> mismo resultado visual, más rápido).
  final smoothed = rawContours
      .map((c) => _chaikinSmooth(c, iterations: 2))
      .toList();

  // Longitud real: esqueleto + camino por la curva. Usamos SOLO el
  // contorno más grande (el mismo que ya se usaba para el área).
  final lengthPx = _skeletonLength(mask8u) ?? _minAreaRectLength(contours[0]);

  mask8u.dispose();

  return _MaskDecodeResult(smoothed, lengthPx);
}

/// Suaviza un polígono cerrado cortando sus esquinas repetidamente
/// (algoritmo de Chaikin). 2 iteraciones suele ser suficiente para
/// quitar el aspecto de escalera sin "derretir" la forma real.
List<Offset> _chaikinSmooth(List<Offset> points, {int iterations = 2}) {
  var pts = points;
  for (int iter = 0; iter < iterations; iter++) {
    if (pts.length < 3) break;
    final n = pts.length;
    final next = <Offset>[];
    for (int i = 0; i < n; i++) {
      final p0 = pts[i];
      final p1 = pts[(i + 1) % n];
      next.add(
        Offset(0.75 * p0.dx + 0.25 * p1.dx, 0.75 * p0.dy + 0.25 * p1.dy),
      );
      next.add(
        Offset(0.25 * p0.dx + 0.75 * p1.dx, 0.25 * p0.dy + 0.75 * p1.dy),
      );
    }
    pts = next;
  }
  return pts;
}

/// Esqueletiza `mask8u` (Zhang-Suen vía OpenCV) y camina por el
/// esqueleto sumando distancias, igual que `longitud_por_esqueleto`
/// en el script de Python. Devuelve null si `ximgproc` no está
/// disponible o el esqueleto sale degenerado (menos de 2 píxeles).
double? _skeletonLength(cv.Mat mask8u) {
  try {
    final skeleton = cv.ximgproc.thinning(mask8u);

    final rows = skeleton.toList();
    skeleton.dispose();

    int encode(int x, int y) => y * 1000000 + x;

    final pointSet = <int>{};
    final coords = <List<int>>[];

    for (int y = 0; y < rows.length; y++) {
      final row = rows[y] as List;
      for (int x = 0; x < row.length; x++) {
        final cell = row[x];
        final value = (cell is List) ? cell[0] : cell;
        if ((value as num) > 0) {
          pointSet.add(encode(x, y));
          coords.add([x, y]);
        }
      }
    }

    if (coords.length < 2) return null;

    const neighbors8 = [
      [-1, -1],
      [-1, 0],
      [-1, 1],
      [0, -1],
      [0, 1],
      [1, -1],
      [1, 0],
      [1, 1],
    ];

    int countNeighbors(int x, int y) {
      int n = 0;
      for (final d in neighbors8) {
        if (pointSet.contains(encode(x + d[0], y + d[1]))) n++;
      }
      return n;
    }

    List<int>? start;
    for (final c in coords) {
      if (countNeighbors(c[0], c[1]) == 1) {
        start = c;
        break;
      }
    }
    start ??= coords.first;

    final visited = <int>{encode(start[0], start[1])};
    var current = start;
    double length = 0;

    while (true) {
      List<int>? next;
      for (final d in neighbors8) {
        final nx = current[0] + d[0];
        final ny = current[1] + d[1];
        final key = encode(nx, ny);
        if (pointSet.contains(key) && !visited.contains(key)) {
          next = [nx, ny];
          break;
        }
      }
      if (next == null) break;
      final dx = (next[0] - current[0]).toDouble();
      final dy = (next[1] - current[1]).toDouble();
      length += math.sqrt(dx * dx + dy * dy);
      visited.add(encode(next[0], next[1]));
      current = next;
    }

    return length;
  } catch (e) {
    debugPrint(
      'ximgproc.thinning no disponible, usando aproximación por '
      'minAreaRect: $e',
    );
    return null;
  }
}

/// Reserva si `ximgproc` no está disponible: lado más largo del
/// rectángulo rotado que envuelve el contorno (línea recta, la misma
/// aproximación de respaldo que usamos en el script de Python).
double _minAreaRectLength(cv.VecPoint contour) {
  final rect = cv.minAreaRect(contour);
  return math.max(rect.size.width, rect.size.height);
}

double _shoelaceArea(List<Offset> poly) {
  double sum = 0;
  for (int i = 0; i < poly.length; i++) {
    final p1 = poly[i];
    final p2 = poly[(i + 1) % poly.length];
    sum += p1.dx * p2.dy - p2.dx * p1.dy;
  }
  return sum.abs() / 2;
}

List<_RawDetection> _nmsPerClass(List<_RawDetection> candidates) {
  final result = <_RawDetection>[];
  final byClass = <int, List<_RawDetection>>{};
  for (final c in candidates) {
    byClass.putIfAbsent(c.classId, () => []).add(c);
  }

  byClass.forEach((classId, list) {
    list.sort((a, b) => b.confidence.compareTo(a.confidence));
    final active = List<bool>.filled(list.length, true);
    for (int i = 0; i < list.length; i++) {
      if (!active[i]) continue;
      result.add(list[i]);
      for (int j = i + 1; j < list.length; j++) {
        if (!active[j]) continue;
        if (_iou(list[i], list[j]) > _nmsIouThreshold) {
          active[j] = false;
        }
      }
    }
  });

  return result;
}

double _iou(_RawDetection a, _RawDetection b) {
  final interX1 = math.max(a.x1, b.x1);
  final interY1 = math.max(a.y1, b.y1);
  final interX2 = math.min(a.x2, b.x2);
  final interY2 = math.min(a.y2, b.y2);

  final interW = math.max(0.0, interX2 - interX1);
  final interH = math.max(0.0, interY2 - interY1);
  final interArea = interW * interH;

  final areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
  final areaB = (b.x2 - b.x1) * (b.y2 - b.y1);

  final unionArea = areaA + areaB - interArea;
  if (unionArea <= 0) return 0;
  return interArea / unionArea;
}

class _MaskDecodeResult {
  final List<List<Offset>> contours;
  final double? lengthPx;
  const _MaskDecodeResult(this.contours, this.lengthPx);
}

class _RawDetection {
  final int classId;
  final double confidence;
  final double x1, y1, x2, y2;
  final int queryIndex;

  _RawDetection({
    required this.classId,
    required this.confidence,
    required this.x1,
    required this.y1,
    required this.x2,
    required this.y2,
    required this.queryIndex,
  });
}
