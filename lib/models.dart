import 'dart:io';
import 'dart:ui' show Offset;

/// Una detección individual (una larva/pupa/etc. encontrada en la foto).
class DetectedObject {
  final int classId;
  final String className;
  final double confidence;
  double? lengthPx; // longitud en píxeles, calculada al decodificar
  double? lengthMm; // longitud en mm, calculada al calibrar con la moneda

  /// Coordenadas en píxeles, relativas a la imagen ORIGINAL (no la
  /// redimensionada a 504x504 que se le da al modelo).
  final double x1, y1, x2, y2;

  /// Contorno(s) de la máscara de segmentación, en el mismo sistema de
  /// coordenadas que x1..y2 (píxeles de la imagen original). Si el
  /// modelo no trae salida de máscaras, queda vacío y la UI cae de
  /// vuelta a dibujar la caja delimitadora.
  ///
  /// Normalmente solo tiene un contorno (el más grande encontrado al
  /// binarizar la máscara); se deja como lista de listas por si en el
  /// futuro se quiere soportar más de un blob por objeto.
  final List<List<Offset>> maskContours;

  /// Tamaño estimado del objeto, en milímetros (diámetro equivalente
  /// calculado a partir del área de su máscara). `null` si no se pudo
  /// calibrar la escala (no se detectó una moneda de referencia en la
  /// foto).
  double? sizeMm;

  DetectedObject({
    required this.classId,
    required this.className,
    required this.confidence,
    required this.x1,
    required this.y1,
    required this.x2,
    required this.y2,
    this.maskContours = const [],
    this.sizeMm,
    this.lengthMm,
    this.lengthPx,
  });

  double get widthPx => x2 - x1;
  double get heightPx => y2 - y1;
}

/// Conteo de una clase específica dentro de una muestra. `count` es
/// mutable a propósito, para permitir la edición manual desde la UI.
class DetectionCount {
  final String className;
  int count;
  DetectionCount({required this.className, required this.count});
}

/// Resultado de ubicar una moneda de referencia en la foto, usado para
/// convertir tamaños de píxeles a milímetros.
class CoinCalibration {
  final double centerX, centerY;
  final double radiusPx;
  final double coinDiameterMm;

  /// Píxeles por milímetro, derivado del diámetro detectado de la
  /// moneda vs. su diámetro real conocido.
  final double pxPerMm;

  CoinCalibration({
    required this.centerX,
    required this.centerY,
    required this.radiusPx,
    required this.coinDiameterMm,
  }) : pxPerMm = (radiusPx * 2) / coinDiameterMm;
}

/// Resultado crudo de correr el modelo sobre una imagen.
class AnalysisOutput {
  final List<DetectedObject> detections;
  final List<DetectionCount> counts;
  final CoinCalibration? calibration;
  AnalysisOutput({
    required this.detections,
    required this.counts,
    this.calibration,
  });
}

/// Un registro/"bloque" que se guarda en la lista principal de la app:
/// una foto ya analizada, con su conteo, sus detecciones y (si se pudo)
/// la calibración de escala usada para estimar tamaños.
class AnalysisRecord {
  final String id;
  String label;
  final DateTime timestamp;
  final File imageFile;
  final int imageWidth;
  final int imageHeight;
  final List<DetectedObject> detections;
  List<DetectionCount> counts;
  CoinCalibration? calibration; // <-- antes era `final`

  AnalysisRecord({
    required this.id,
    required this.label,
    required this.timestamp,
    required this.imageFile,
    required this.imageWidth,
    required this.imageHeight,
    required this.detections,
    required this.counts,
    this.calibration,
  });

  int get total => counts.fold(0, (sum, c) => sum + c.count);
}
