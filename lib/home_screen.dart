import 'dart:io';

import 'package:flutter/material.dart';
import 'package:image/image.dart' as img;
import 'package:image_picker/image_picker.dart';
import 'package:intl/intl.dart';
import 'package:uuid/uuid.dart';
import 'package:path_provider/path_provider.dart';

import 'camera_screen.dart';
import 'models.dart';
import 'rfdetr_service.dart';
import 'dart:ui' as ui;
import 'coin_detector.dart';
import 'mask_renderer.dart';
import 'dart:math' as math;
import 'dart:typed_data';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  final _service = RfDetrService();
  final _uuid = const Uuid();
  final List<AnalysisRecord> _records = [];

  bool _loadingModel = true;
  bool _analyzing = false;
  String? _modelError;

  @override
  void initState() {
    super.initState();
    _loadModel();
  }

  Future<void> _loadModel() async {
    try {
      await _service.loadModel();
      setState(() => _loadingModel = false);
    } catch (e) {
      setState(() {
        _modelError = e.toString();
        _loadingModel = false;
      });
    }
  }

  Future<void> _onAddPressed() async {
    if (!_service.isReady) return;

    final source = await showModalBottomSheet<ImageSource>(
      context: context,
      builder: (context) => SafeArea(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            ListTile(
              leading: const Icon(Icons.camera_alt),
              title: const Text('Tomar foto'),
              onTap: () => Navigator.pop(context, ImageSource.camera),
            ),
            ListTile(
              leading: const Icon(Icons.photo_library),
              title: const Text('Elegir de la galería'),
              onTap: () => Navigator.pop(context, ImageSource.gallery),
            ),
          ],
        ),
      ),
    );
    if (source == null) return;

    File? file;
    if (source == ImageSource.camera) {
      file = await Navigator.of(
        context,
      ).push<File>(MaterialPageRoute(builder: (_) => const CameraScreen()));
    } else {
      final picked = await ImagePicker().pickImage(source: ImageSource.gallery);
      if (picked != null) file = File(picked.path);
    }
    if (file == null) return;

    await _analyzeAndAddRecord(file);
  }

  Future<void> _analyzeAndAddRecord(File file) async {
    setState(() => _analyzing = true);
    try {
      final bytes = await file.readAsBytes();
      var decoded = img.decodeImage(bytes);
      if (decoded == null) {
        throw Exception('No se pudo leer la imagen seleccionada.');
      }
      decoded = img.bakeOrientation(decoded);

      // Acotamos la resolución de TRABAJO (no solo la del modelo, que ya
      // se reduce a 504x504 dentro de RfDetrService). Lo que de verdad
      // pesa es el decode de máscaras + findContours + esqueleto, que se
      // hacen a la resolución de "decoded" -> si viene una foto de
      // cámara de 12-48MP sin recortar, cada larva detectada arrastra
      // ese costo. 1600px de lado más largo es más que suficiente para
      // ver el detalle de una larva y mantiene todo rápido.
      decoded = _capWorkingResolution(decoded, 1600);

      // Guardamos la versión YA acotada (no el archivo gigante original)
      // para que el análisis, las medidas y la visualización usen
      // siempre la misma resolución -> evita desalinear contornos.
      final workDir = await getApplicationDocumentsDirectory();
      final workingPath =
          '${workDir.path}/analisis_${DateTime.now().millisecondsSinceEpoch}.jpg';
      final workingFile = File(workingPath);
      await workingFile.writeAsBytes(img.encodeJpg(decoded, quality: 92));

      final output = await _service.analyze(decoded);

      final record = AnalysisRecord(
        id: _uuid.v4(),
        label: 'Muestra ${_records.length + 1}',
        timestamp: DateTime.now(),
        imageFile: workingFile,
        imageWidth: decoded.width,
        imageHeight: decoded.height,
        detections: output.detections,
        counts: output.counts,
        calibration: output.calibration,
      );

      setState(() => _records.insert(0, record));
    } catch (e, stackTrace) {
      debugPrint('=== ERROR AL ANALIZAR ===');
      debugPrint(e.toString());
      debugPrint(stackTrace.toString());
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Error al analizar la imagen: $e')),
        );
      }
    } finally {
      if (mounted) setState(() => _analyzing = false);
    }
  }

  img.Image _capWorkingResolution(img.Image image, int maxSide) {
    final longest = math.max(image.width, image.height);
    if (longest <= maxSide) return image;
    return image.width >= image.height
        ? img.copyResize(image, width: maxSide)
        : img.copyResize(image, height: maxSide);
  }

  void _deleteRecord(AnalysisRecord record) {
    setState(() => _records.removeWhere((r) => r.id == record.id));
  }

  Future<void> _editRecord(AnalysisRecord record) async {
    final labelController = TextEditingController(text: record.label);
    final countsCopy = record.counts
        .map((c) => DetectionCount(className: c.className, count: c.count))
        .toList();

    final saved = await showModalBottomSheet<bool>(
      context: context,
      isScrollControlled: true,
      builder: (context) {
        return Padding(
          padding: EdgeInsets.only(
            left: 16,
            right: 16,
            top: 16,
            bottom: MediaQuery.of(context).viewInsets.bottom + 16,
          ),
          child: StatefulBuilder(
            builder: (context, setModalState) {
              return SingleChildScrollView(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.stretch,
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    const Text(
                      'Editar muestra',
                      style: TextStyle(
                        fontSize: 18,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                    const SizedBox(height: 12),
                    TextField(
                      controller: labelController,
                      decoration: const InputDecoration(
                        labelText: 'Nombre de la muestra',
                        border: OutlineInputBorder(),
                      ),
                    ),
                    const SizedBox(height: 16),
                    const Text(
                      'Conteo por clase',
                      style: TextStyle(fontWeight: FontWeight.bold),
                    ),
                    ...countsCopy.map(
                      (c) => Padding(
                        padding: const EdgeInsets.symmetric(vertical: 4),
                        child: Row(
                          children: [
                            Expanded(child: Text(c.className)),
                            IconButton(
                              icon: const Icon(Icons.remove_circle_outline),
                              onPressed: () {
                                setModalState(() {
                                  if (c.count > 0) c.count--;
                                });
                              },
                            ),
                            SizedBox(
                              width: 28,
                              child: Text(
                                '${c.count}',
                                textAlign: TextAlign.center,
                                style: const TextStyle(fontSize: 16),
                              ),
                            ),
                            IconButton(
                              icon: const Icon(Icons.add_circle_outline),
                              onPressed: () {
                                setModalState(() => c.count++);
                              },
                            ),
                          ],
                        ),
                      ),
                    ),
                    const SizedBox(height: 16),
                    Row(
                      children: [
                        Expanded(
                          child: OutlinedButton(
                            onPressed: () => Navigator.pop(context, false),
                            child: const Text('Cancelar'),
                          ),
                        ),
                        const SizedBox(width: 12),
                        Expanded(
                          child: FilledButton(
                            onPressed: () => Navigator.pop(context, true),
                            child: const Text('Guardar'),
                          ),
                        ),
                      ],
                    ),
                  ],
                ),
              );
            },
          ),
        );
      },
    );

    if (saved == true) {
      setState(() {
        record.label = labelController.text.trim().isEmpty
            ? record.label
            : labelController.text.trim();
        record.counts = countsCopy;
      });
    }
  }

  void _showDetail(AnalysisRecord record) {
    Navigator.of(context).push(
      MaterialPageRoute(
        fullscreenDialog: true,
        builder: (_) => _RecordDetailScreen(record: record),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Conteo de larvas BSF')),
      body: _loadingModel
          ? const Center(child: CircularProgressIndicator())
          : _modelError != null
          ? Center(
              child: Padding(
                padding: const EdgeInsets.all(24),
                child: Text(
                  'No se pudo cargar el modelo:\n$_modelError',
                  textAlign: TextAlign.center,
                ),
              ),
            )
          : Stack(
              children: [
                _records.isEmpty
                    ? const Center(
                        child: Text('Toma una foto para empezar a contar'),
                      )
                    : ListView.builder(
                        padding: const EdgeInsets.all(12),
                        itemCount: _records.length,
                        itemBuilder: (context, index) {
                          final record = _records[index];
                          return Card(
                            margin: const EdgeInsets.only(bottom: 12),
                            child: ListTile(
                              onTap: () => _showDetail(record),
                              leading: ClipRRect(
                                borderRadius: BorderRadius.circular(8),
                                child: Image.file(
                                  record.imageFile,
                                  width: 56,
                                  height: 56,
                                  fit: BoxFit.cover,
                                ),
                              ),
                              title: Text(record.label),
                              subtitle: Text(
                                'Total: ${record.total} · '
                                '${DateFormat('dd/MM HH:mm').format(record.timestamp)}',
                              ),
                              trailing: Row(
                                mainAxisSize: MainAxisSize.min,
                                children: [
                                  IconButton(
                                    icon: const Icon(Icons.edit_outlined),
                                    onPressed: () => _editRecord(record),
                                  ),
                                  IconButton(
                                    icon: const Icon(Icons.delete_outline),
                                    onPressed: () async {
                                      final confirm = await showDialog<bool>(
                                        context: context,
                                        builder: (context) => AlertDialog(
                                          title: const Text('Eliminar muestra'),
                                          content: Text(
                                            '¿Eliminar "${record.label}"? '
                                            'Esta acción no se puede deshacer.',
                                          ),
                                          actions: [
                                            TextButton(
                                              onPressed: () =>
                                                  Navigator.pop(context, false),
                                              child: const Text('Cancelar'),
                                            ),
                                            TextButton(
                                              onPressed: () =>
                                                  Navigator.pop(context, true),
                                              child: const Text('Eliminar'),
                                            ),
                                          ],
                                        ),
                                      );
                                      if (confirm == true)
                                        _deleteRecord(record);
                                    },
                                  ),
                                ],
                              ),
                            ),
                          );
                        },
                      ),
                if (_analyzing)
                  Container(
                    color: Colors.black45,
                    child: const Center(
                      child: Card(
                        child: Padding(
                          padding: EdgeInsets.all(24),
                          child: Column(
                            mainAxisSize: MainAxisSize.min,
                            children: [
                              CircularProgressIndicator(),
                              SizedBox(height: 12),
                              Text('Analizando imagen...'),
                            ],
                          ),
                        ),
                      ),
                    ),
                  ),
              ],
            ),
      floatingActionButton: FloatingActionButton.extended(
        onPressed: (!_loadingModel && _modelError == null && !_analyzing)
            ? _onAddPressed
            : null,
        icon: const Icon(Icons.camera_alt),
        label: const Text('Tomar foto'),
      ),
    );
  }
}

/// Hoja de detalle de una muestra: foto con las máscaras superpuestas,
/// interactiva (tocar una larva la resalta y opaca el resto; tocar un
/// espacio vacío vuelve todo a la normalidad) y el resumen de conteo.

class _RecordDetailScreen extends StatefulWidget {
  final AnalysisRecord record;
  const _RecordDetailScreen({required this.record});

  @override
  State<_RecordDetailScreen> createState() => _RecordDetailScreenState();
}

class _RecordDetailScreenState extends State<_RecordDetailScreen> {
  int? _selectedIndex;
  bool _calibrationMode = false;
  Offset? _calCenterImg;
  double? _calRadiusImg;
  final _viewerController = TransformationController();

  img.Image? _decodedImage;
  Uint8List? _renderedBytes;

  @override
  void initState() {
    super.initState();
    _loadAndRender();
  }

  Future<void> _loadAndRender() async {
    final bytes = await widget.record.imageFile.readAsBytes();
    final decoded = img.decodeImage(bytes);
    if (decoded == null) return;
    _decodedImage = decoded;
    _rebuildRender();
  }

  /// Vuelve a "hornear" el relleno + contorno + etiqueta con OpenCV
  /// (mask_renderer.dart) sobre la foto ya decodificada. Se llama cada
  /// vez que cambia la selección, NO en cada frame de arrastre de la
  /// calibración (ahí seguimos usando el círculo vectorial, más ágil).
  void _rebuildRender() {
    final decoded = _decodedImage;
    if (decoded == null) return;
    try {
      final bytes = MaskRenderer.render(
        baseImage: decoded,
        detections: widget.record.detections,
        highlightIndex: _selectedIndex,
      );
      setState(() => _renderedBytes = bytes);
    } catch (e) {
      debugPrint('Error al renderizar máscaras con OpenCV: $e');
      // Si falla, seguimos mostrando la foto simple (ver build()).
    }
  }

  void _handleTap(Offset localPos, Size boxSize) {
    if (_calibrationMode) return; // en calibración se usa el círculo, no el tap
    final record = widget.record;
    final scaleX = boxSize.width / record.imageWidth;
    final scaleY = boxSize.height / record.imageHeight;
    final imgX = localPos.dx / scaleX;
    final imgY = localPos.dy / scaleY;

    int? hit;
    for (int i = record.detections.length - 1; i >= 0; i--) {
      if (_pointInDetection(record.detections[i], imgX, imgY)) {
        hit = i;
        break;
      }
    }
    setState(() => _selectedIndex = hit);
    _rebuildRender();
  }

  bool _pointInDetection(DetectedObject d, double x, double y) {
    if (d.maskContours.isNotEmpty) {
      final path = Path()..addPolygon(d.maskContours.first, true);
      return path.contains(Offset(x, y));
    }
    return x >= d.x1 && x <= d.x2 && y >= d.y1 && y <= d.y2;
  }

  void _startCalibration() {
    final record = widget.record;
    final minDim = math.min(record.imageWidth, record.imageHeight).toDouble();
    final existing = record.calibration;
    setState(() {
      _calibrationMode = true;
      _selectedIndex = null;
      _calCenterImg = existing != null
          ? Offset(existing.centerX, existing.centerY)
          : Offset(record.imageWidth / 2, record.imageHeight / 2);
      _calRadiusImg = existing?.radiusPx ?? minDim * 0.08;
    });
    _viewerController.value =
        Matrix4.identity(); // vista limpia, sin zoom previo
  }

  void _cancelCalibration() {
    setState(() {
      _calibrationMode = false;
      _calCenterImg = null;
      _calRadiusImg = null;
    });
  }

  Future<void> _confirmCalibration() async {
    final diameterMm = await _askCoinDiameterMm();
    if (diameterMm == null || diameterMm <= 0)
      return; // canceló, sigue calibrando

    final calibration = CoinCalibration(
      centerX: _calCenterImg!.dx,
      centerY: _calCenterImg!.dy,
      radiusPx: _calRadiusImg!,
      coinDiameterMm: diameterMm,
    );
    RfDetrService.applyCalibration(widget.record.detections, calibration);

    setState(() {
      widget.record.calibration = calibration;
      _calibrationMode = false;
      _calCenterImg = null;
      _calRadiusImg = null;
    });
  }

  Future<double?> _askCoinDiameterMm() {
    final controller = TextEditingController(
      text: CoinDetector.defaultCoinDiameterMm.toStringAsFixed(1),
    );
    return showDialog<double>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Diámetro de la moneda'),
        content: TextField(
          controller: controller,
          autofocus: true,
          keyboardType: const TextInputType.numberWithOptions(decimal: true),
          decoration: const InputDecoration(
            suffixText: 'mm',
            helperText: 'Diámetro real de la moneda que ajustaste',
          ),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Cancelar'),
          ),
          FilledButton(
            onPressed: () {
              final value = double.tryParse(
                controller.text.replaceAll(',', '.'),
              );
              Navigator.pop(context, value);
            },
            child: const Text('Confirmar'),
          ),
        ],
      ),
    );
  }

  List<Widget> _buildCalibrationHandles(double scaleX, double scaleY) {
    const handleSize = 28.0;
    final center = _calCenterImg!;
    final radius = _calRadiusImg!;
    final centerWidget = Offset(center.dx * scaleX, center.dy * scaleY);
    final edgeWidget = Offset(
      (center.dx + radius) * scaleX,
      center.dy * scaleY,
    );

    Widget handle(
      Offset pos,
      Color color,
      void Function(DragUpdateDetails) onDrag,
    ) {
      return Positioned(
        left: pos.dx - handleSize / 2,
        top: pos.dy - handleSize / 2,
        child: GestureDetector(
          onPanUpdate: onDrag,
          child: Container(
            width: handleSize,
            height: handleSize,
            decoration: BoxDecoration(
              shape: BoxShape.circle,
              color: color,
              border: Border.all(color: Colors.black87, width: 2),
            ),
          ),
        ),
      );
    }

    return [
      // Arrastra TODO el círculo (mover).
      handle(centerWidget, Colors.yellow, (details) {
        setState(() {
          _calCenterImg = Offset(
            _calCenterImg!.dx + details.delta.dx / scaleX,
            _calCenterImg!.dy + details.delta.dy / scaleY,
          );
        });
      }),
      // Arrastra para cambiar el radio (hacia afuera = más grande).
      handle(edgeWidget, Colors.orangeAccent, (details) {
        setState(() {
          final maxR =
              math.min(widget.record.imageWidth, widget.record.imageHeight) / 2;
          _calRadiusImg = (_calRadiusImg! + details.delta.dx / scaleX).clamp(
            5.0,
            maxR,
          );
        });
      }),
    ];
  }

  @override
  Widget build(BuildContext context) {
    final record = widget.record;
    final selected = _selectedIndex != null
        ? record.detections[_selectedIndex!]
        : null;

    return Scaffold(
      appBar: AppBar(title: Text(record.label)),
      body: Column(
        children: [
          Expanded(
            child: Container(
              color: Colors.black,
              child: Center(
                child: AspectRatio(
                  aspectRatio: record.imageWidth / record.imageHeight,
                  child: LayoutBuilder(
                    builder: (context, constraints) {
                      final boxSize = constraints.biggest;
                      final scaleX = boxSize.width / record.imageWidth;
                      final scaleY = boxSize.height / record.imageHeight;
                      return InteractiveViewer(
                        transformationController: _viewerController,
                        minScale: 1,
                        maxScale: 8,
                        boundaryMargin: const EdgeInsets.all(40),
                        panEnabled: !_calibrationMode,
                        scaleEnabled: !_calibrationMode,
                        child: GestureDetector(
                          behavior: HitTestBehavior.opaque,
                          onTapUp: (d) => _handleTap(d.localPosition, boxSize),
                          child: SizedBox(
                            width: boxSize.width,
                            height: boxSize.height,
                            child: Stack(
                              fit: StackFit.expand,
                              children: [
                                Image.file(record.imageFile, fit: BoxFit.cover),
                                CustomPaint(
                                  painter: _MaskPainter(
                                    record: record,
                                    selectedIndex: _selectedIndex,
                                    calibrationMode: _calibrationMode,
                                    calibCenterImg: _calCenterImg,
                                    calibRadiusImg: _calRadiusImg,
                                  ),
                                ),
                                if (_calibrationMode &&
                                    _calCenterImg != null &&
                                    _calRadiusImg != null)
                                  ..._buildCalibrationHandles(scaleX, scaleY),
                              ],
                            ),
                          ),
                        ),
                      );
                    },
                  ),
                ),
              ),
            ),
          ),
          SafeArea(
            top: false,
            child: ConstrainedBox(
              constraints: BoxConstraints(
                maxHeight: MediaQuery.of(context).size.height * 0.38,
              ),
              child: SingleChildScrollView(
                padding: const EdgeInsets.all(16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.stretch,
                  children: [
                    Text(
                      DateFormat('dd/MM/yyyy HH:mm').format(record.timestamp),
                      style: TextStyle(color: Colors.grey[600]),
                    ),
                    const SizedBox(height: 12),
                    Row(
                      children: [
                        Expanded(
                          child: Text(
                            _calibrationMode
                                ? 'Ajustá el círculo amarillo sobre la moneda '
                                      '(arrastrá el centro para moverlo, el '
                                      'punto naranja para cambiar el tamaño).'
                                : (record.calibration != null
                                      ? 'Escala: ${record.calibration!.pxPerMm.toStringAsFixed(1)} px/mm '
                                            '(moneda de ${record.calibration!.coinDiameterMm.toStringAsFixed(1)} mm)'
                                      : 'Sin moneda de referencia: los tamaños en mm no están disponibles.'),
                            style: TextStyle(
                              fontSize: 12,
                              color: Colors.grey[600],
                              fontStyle: FontStyle.italic,
                            ),
                          ),
                        ),
                        if (_calibrationMode) ...[
                          TextButton(
                            onPressed: _cancelCalibration,
                            child: const Text('Cancelar'),
                          ),
                          FilledButton(
                            onPressed: _confirmCalibration,
                            child: const Text('Confirmar'),
                          ),
                        ] else
                          TextButton.icon(
                            onPressed: _startCalibration,
                            icon: const Icon(
                              Icons.monetization_on_outlined,
                              size: 18,
                            ),
                            label: const Text('Calibrar moneda'),
                          ),
                      ],
                    ),
                    const SizedBox(height: 12),
                    if (selected != null)
                      Card(
                        color: Theme.of(context).colorScheme.secondaryContainer,
                        child: Padding(
                          padding: const EdgeInsets.all(12),
                          child: Row(
                            children: [
                              Expanded(
                                child: Column(
                                  crossAxisAlignment: CrossAxisAlignment.start,
                                  children: [
                                    Text(
                                      selected.className,
                                      style: const TextStyle(
                                        fontWeight: FontWeight.bold,
                                      ),
                                    ),
                                    Text(
                                      'Confianza: ${(selected.confidence * 100).toStringAsFixed(0)}%',
                                    ),
                                    Text(
                                      selected.lengthMm != null
                                          ? 'Longitud: ${selected.lengthMm!.toStringAsFixed(1)} mm'
                                          : (selected.lengthPx != null
                                                ? 'Longitud: ${selected.lengthPx!.toStringAsFixed(0)} px (sin calibrar)'
                                                : 'Longitud: no disponible'),
                                    ),
                                  ],
                                ),
                              ),
                              IconButton(
                                icon: const Icon(Icons.close),
                                onPressed: () =>
                                    setState(() => _selectedIndex = null),
                              ),
                            ],
                          ),
                        ),
                      )
                    else
                      Text(
                        'Toca una larva/pupa en la foto para ver sus detalles.',
                        style: TextStyle(fontSize: 12, color: Colors.grey[600]),
                      ),
                    const SizedBox(height: 16),
                    Text(
                      'Total: ${record.total}',
                      style: const TextStyle(
                        fontSize: 16,
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                    const SizedBox(height: 8),
                    ...record.counts.map(
                      (c) => Padding(
                        padding: const EdgeInsets.symmetric(vertical: 2),
                        child: Row(
                          mainAxisAlignment: MainAxisAlignment.spaceBetween,
                          children: [
                            Text(c.className),
                            Text(
                              '${c.count}',
                              style: const TextStyle(
                                fontWeight: FontWeight.bold,
                              ),
                            ),
                          ],
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}

/// Dibuja las máscaras de detección (o la caja, si una detección no
/// tiene máscara) sobre la foto en el modal de detalle. Si hay una
/// detección seleccionada, la resalta y opaca las demás.
class _MaskPainter extends CustomPainter {
  final AnalysisRecord record;
  final int? selectedIndex;
  final bool calibrationMode;
  final Offset? calibCenterImg;
  final double? calibRadiusImg;

  _MaskPainter({
    required this.record,
    required this.selectedIndex,
    this.calibrationMode = false,
    this.calibCenterImg,
    this.calibRadiusImg,
  });
  static const colors = {
    1: Colors.orange,
    2: Colors.blueAccent,
    3: Colors.purpleAccent,
    4: Colors.green,
  };

  @override
  void paint(Canvas canvas, Size size) {
    final scaleX = size.width / record.imageWidth;
    final scaleY = size.height / record.imageHeight;
    final hasSelection = selectedIndex != null;

    final textPainter = TextPainter(textDirection: ui.TextDirection.ltr);

    for (int i = 0; i < record.detections.length; i++) {
      final d = record.detections[i];
      final isSelected = selectedIndex == i;
      final isDimmed = calibrationMode || (hasSelection && !isSelected);
      final baseColor = colors[d.classId] ?? Colors.redAccent;

      final fillOpacity = isDimmed ? 0.06 : (isSelected ? 0.55 : 0.35);
      final strokeOpacity = isDimmed ? 0.15 : 1.0;
      final strokeWidth = isSelected ? 3.0 : 1.5;

      final fillPaint = Paint()
        ..color = baseColor.withOpacity(fillOpacity)
        ..style = PaintingStyle.fill;
      final strokePaint = Paint()
        ..color = baseColor.withOpacity(strokeOpacity)
        ..style = PaintingStyle.stroke
        ..strokeWidth = strokeWidth;

      if (d.maskContours.isNotEmpty) {
        final path = Path();
        for (final contour in d.maskContours) {
          final scaled = contour
              .map((o) => Offset(o.dx * scaleX, o.dy * scaleY))
              .toList();
          path.addPolygon(scaled, true);
        }
        canvas.drawPath(path, fillPaint);
        canvas.drawPath(path, strokePaint);
      } else {
        // Sin máscara (p. ej. el modelo no trajo salida de segmentación
        // o falló al decodificarla para esta query): caemos de vuelta
        // a la caja delimitadora.
        final rect = Rect.fromLTRB(
          d.x1 * scaleX,
          d.y1 * scaleY,
          d.x2 * scaleX,
          d.y2 * scaleY,
        );
        canvas.drawRect(rect, fillPaint);
        canvas.drawRect(rect, strokePaint);
      }

      // Etiqueta de clase + confianza, siempre visible (igual que en la
      // referencia de Python), y con el tamaño en mm cuando el objeto
      // está seleccionado y hay calibración disponible.
      final labelText = isSelected && d.lengthMm != null
          ? '${i + 1}. ${d.className} ${d.confidence.toStringAsFixed(2)} · '
                '${d.lengthMm!.toStringAsFixed(1)}mm'
          : '${i + 1}. ${d.className} ${d.confidence.toStringAsFixed(2)}';

      textPainter.text = TextSpan(
        text: labelText,
        style: TextStyle(
          color: Colors.white.withOpacity(isDimmed ? 0.35 : 1.0),
          fontSize: isSelected ? 11 : 9,
          fontWeight: isSelected ? FontWeight.bold : FontWeight.normal,
          backgroundColor: baseColor.withOpacity(isDimmed ? 0.15 : 0.9),
        ),
      );
      textPainter.layout();
      textPainter.paint(
        canvas,
        Offset(d.x1 * scaleX, (d.y1 * scaleY - 14).clamp(0, size.height)),
      );
    }

    // Al final del paint(), reemplaza el bloque del círculo amarillo:
    if (calibrationMode && calibCenterImg != null && calibRadiusImg != null) {
      final c = Offset(
        calibCenterImg!.dx * scaleX,
        calibCenterImg!.dy * scaleY,
      );
      final r =
          calibRadiusImg! *
          scaleX; // scaleX≈scaleY: el AspectRatio ya respeta la proporción real
      canvas.drawCircle(
        c,
        r,
        Paint()
          ..color = Colors.yellow.withOpacity(0.25)
          ..style = PaintingStyle.fill,
      );
      canvas.drawCircle(
        c,
        r,
        Paint()
          ..color = Colors.yellow
          ..style = PaintingStyle.stroke
          ..strokeWidth = 2,
      );
    } else if (record.calibration != null) {
      final cal = record.calibration!;
      canvas.drawCircle(
        Offset(cal.centerX * scaleX, cal.centerY * scaleY),
        cal.radiusPx * scaleX,
        Paint()
          ..color = Colors.yellow
          ..style = PaintingStyle.stroke
          ..strokeWidth = 2,
      );
    }
  }

  @override
  bool shouldRepaint(covariant _MaskPainter oldDelegate) =>
      oldDelegate.selectedIndex != selectedIndex ||
      oldDelegate.record != record ||
      oldDelegate.calibrationMode != calibrationMode ||
      oldDelegate.calibCenterImg != calibCenterImg ||
      oldDelegate.calibRadiusImg != calibRadiusImg;
}
