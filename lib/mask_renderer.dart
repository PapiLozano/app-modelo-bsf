import 'dart:typed_data';
import 'dart:ui' show Offset;

import 'package:image/image.dart' as img;
import 'package:opencv_core/opencv.dart' as cv;

import 'models.dart';

/// Compone la imagen final (relleno + contorno + etiqueta numerada) con
/// las MISMAS primitivas de dibujo que usa OpenCV en Python
/// (fillPoly + addWeighted para el relleno translúcido, polylines con
/// antialiasing para el borde, rectangle+putText para la etiqueta),
/// en vez de un `Path` vectorial de Flutter. Esto es lo que realmente
/// da el acabado "raster" liso, igual al de supervision/cv2.
///
/// NOTA: esta es la primera vez que probamos fillPoly/polylines/putText
/// en tu build — si algo no compila o el nombre de un parámetro es
/// distinto en tu versión de `opencv_core`, dime el error exacto y lo
/// ajustamos (el patrón general ya está confirmado con la doc oficial,
/// pero puede haber pequeñas diferencias de firma entre versiones).
class MaskRenderer {
  static const Map<int, (int, int, int)> _colorsBgr = {
    // OpenCV usa BGR, no RGB -> por eso el orden está invertido
    // respecto a los Colors.xxx de Flutter.
    1: (0, 165, 255), // imago  - naranja
    2: (255, 144, 30), // larva  - azul
    3: (211, 85, 186), // prepupa - morado
    4: (34, 139, 34), // pupa   - verde
  };

  /// Devuelve un JPEG (bytes) con todas las detecciones ya "horneadas"
  /// sobre la foto. Si `highlightIndex` no es null, esa detección se ve
  /// resaltada (más opaca) y el resto se atenúa, igual que hacía el
  /// CustomPainter anterior.
  static Uint8List render({
    required img.Image baseImage,
    required List<DetectedObject> detections,
    int? highlightIndex,
  }) {
    final basePng = Uint8List.fromList(img.encodePng(baseImage));
    final mat = cv.imdecode(basePng, cv.IMREAD_COLOR);

    final hasHighlight = highlightIndex != null;

    // 1. Relleno translúcido: se dibuja en una copia y se mezcla con
    //    addWeighted, igual que supervision.MaskAnnotator en Python.
    for (int i = 0; i < detections.length; i++) {
      final d = detections[i];
      if (d.maskContours.isEmpty) continue;

      final isSelected = highlightIndex == i;
      final opacity = hasHighlight ? (isSelected ? 0.55 : 0.12) : 0.35;

      final overlay = mat.clone();
      final colorBgr = _colorsBgr[d.classId] ?? (0, 0, 255);
      final contourPts = _toPoints(d.maskContours.first);
      final polys = cv.VecVecPoint.fromList([contourPts.toList()]);

      cv.fillPoly(
        overlay,
        polys,
        cv.Scalar(
          colorBgr.$1.toDouble(),
          colorBgr.$2.toDouble(),
          colorBgr.$3.toDouble(),
          0,
        ),
        lineType: cv.LINE_AA,
      );

      cv.addWeighted(overlay, opacity, mat, 1 - opacity, 0, dst: mat);
      overlay.dispose();
    }

    // 2. Contorno nítido (sin mezclar, directo sobre la imagen final).
    for (int i = 0; i < detections.length; i++) {
      final d = detections[i];
      if (d.maskContours.isEmpty) continue;

      final isSelected = highlightIndex == i;
      final isDimmed = hasHighlight && !isSelected;
      final colorBgr = _colorsBgr[d.classId] ?? (0, 0, 255);
      final thickness = isSelected ? 3 : 2;

      if (isDimmed) continue; // no dibujamos borde de lo atenuado

      final contourPts = _toPoints(d.maskContours.first);
      final polys = cv.VecVecPoint.fromList([contourPts.toList()]);

      cv.polylines(
        mat,
        polys,
        true, // isClosed
        cv.Scalar(
          colorBgr.$1.toDouble(),
          colorBgr.$2.toDouble(),
          colorBgr.$3.toDouble(),
          0,
        ),
        thickness: thickness,
        lineType: cv.LINE_AA,
      );
    }

    // 3. Etiqueta numerada (caja negra + texto blanco), como en la
    //    referencia de Python.
    for (int i = 0; i < detections.length; i++) {
      final d = detections[i];
      if (hasHighlight && highlightIndex != i)
        continue; // solo la seleccionada, o todas si no hay selección
      final anchorX = d.x1.round();
      final anchorY = (d.y1 - 22).round().clamp(0, mat.rows - 1);

      final text = '${i + 1}';
      cv.rectangle(
        mat,
        cv.Rect(anchorX, anchorY, 26, 22),
        cv.Scalar(0, 0, 0, 0),
        thickness: -1, // relleno
      );
      cv.putText(
        mat,
        text,
        cv.Point(anchorX + 5, anchorY + 17),
        cv.FONT_HERSHEY_SIMPLEX,
        0.6,
        cv.Scalar(255, 255, 255, 0),
        thickness: 2,
        lineType: cv.LINE_AA,
      );
    }

    final resultBytes = cv.imencode('.jpg', mat).$2;
    mat.dispose();
    return resultBytes;
  }

  static cv.VecPoint _toPoints(List<Offset> offsets) {
    return cv.VecPoint.fromList(
      offsets.map((o) => cv.Point(o.dx.round(), o.dy.round())).toList(),
    );
  }
}
