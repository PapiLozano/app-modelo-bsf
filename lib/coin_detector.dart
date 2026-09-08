import 'dart:typed_data';
import 'package:image/image.dart' as img;
import 'package:opencv_core/opencv.dart' as cv;

class CircleCandidate {
  final double cx, cy, r;
  CircleCandidate(this.cx, this.cy, this.r);
}

class CoinDetector {
  static const double defaultCoinDiameterMm = 23.0;
  static const int workingWidth = 640;

  static CircleCandidate? findCoin(
    img.Image original, {
    double minRadiusFraction = 0.03,
    double maxRadiusFraction = 0.25,
  }) {
    final scale = original.width > workingWidth
        ? workingWidth / original.width
        : 1.0;
    final work = scale < 1.0
        ? img.copyResize(original, width: workingWidth)
        : original;

    // Codificamos a PNG en memoria y se lo pasamos a OpenCV ya como
    // escala de grises: evita armar el Mat canal por canal a mano.
    final pngBytes = Uint8List.fromList(img.encodePng(work));
    final gray = cv.imdecode(pngBytes, cv.IMREAD_GRAYSCALE);

    // Blur de mediana: el paso estándar antes de HoughCircles según
    // la propia documentación de OpenCV, saca ruido sin borrar bordes.
    final blurred = cv.medianBlur(gray, 5);
    gray.dispose();

    final w = blurred.cols, h = blurred.rows;
    final minR = (w < h ? w : h) * minRadiusFraction;
    final maxR = (w < h ? w : h) * maxRadiusFraction;

    final circlesMat = cv.HoughCircles(
      blurred,
      cv.HOUGH_GRADIENT,
      1.0, // dp: resolución del acumulador vs la imagen
      h / 8, // minDist entre centros (no esperamos 2 monedas)
      param1: 100, // umbral alto interno de Canny
      param2: 40, // umbral del acumulador: más bajo = más permisivo
      minRadius: minR.round(),
      maxRadius: maxR.round(),
    );
    blurred.dispose();

    if (circlesMat.rows == 0 || circlesMat.cols == 0) {
      circlesMat.dispose();
      return null;
    }

    // Forma (1, N, 3): cada entrada es [cx, cy, r]. OpenCV devuelve
    // el más "circular" primero según el acumulador.
    final rows = circlesMat.toList();
    circlesMat.dispose();

    final best = (rows[0] as List)[0] as List;
    final cx = (best[0] as num).toDouble();
    final cy = (best[1] as num).toDouble();
    final r = (best[2] as num).toDouble();

    return CircleCandidate(cx / scale, cy / scale, r / scale);
  }
}
