import 'dart:io';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:path_provider/path_provider.dart';

class CameraScreen extends StatefulWidget {
  const CameraScreen({super.key});

  @override
  State<CameraScreen> createState() => _CameraScreenState();
}

class _CameraScreenState extends State<CameraScreen> {
  CameraController? _controller;
  Future<void>? _initFuture;
  bool _capturing = false;
  String? _error;

  @override
  void initState() {
    super.initState();
    _setup();
  }

  Future<void> _setup() async {
    try {
      final cameras = await availableCameras();
      if (cameras.isEmpty) {
        setState(() => _error = 'No se encontró ninguna cámara.');
        return;
      }
      final back = cameras.firstWhere(
        (c) => c.lensDirection == CameraLensDirection.back,
        orElse: () => cameras.first,
      );
      _controller = CameraController(
        back,
        ResolutionPreset.high,
        enableAudio: false,
      );
      _initFuture = _controller!.initialize();
      if (mounted) setState(() {});
    } catch (e) {
      if (mounted) setState(() => _error = 'Error de cámara: $e');
    }
  }

  @override
  void dispose() {
    _controller?.dispose();
    super.dispose();
  }

  Future<void> _takePhoto() async {
    if (_controller == null || _capturing) return;
    setState(() => _capturing = true);
    try {
      final xfile = await _controller!.takePicture();
      final dir = await getApplicationDocumentsDirectory();
      final fileName = 'captura_${DateTime.now().millisecondsSinceEpoch}.jpg';
      final savedPath = '${dir.path}/$fileName';
      await File(xfile.path).copy(savedPath);
      if (mounted) Navigator.of(context).pop(File(savedPath));
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(
          context,
        ).showSnackBar(SnackBar(content: Text('Error al tomar la foto: $e')));
      }
    } finally {
      if (mounted) setState(() => _capturing = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    if (_error != null) {
      return Scaffold(
        appBar: AppBar(title: const Text('Tomar foto')),
        body: Center(
          child: Padding(
            padding: const EdgeInsets.all(24),
            child: Text(_error!, textAlign: TextAlign.center),
          ),
        ),
      );
    }

    if (_controller == null || _initFuture == null) {
      return const Scaffold(body: Center(child: CircularProgressIndicator()));
    }

    return Scaffold(
      appBar: AppBar(title: const Text('Tomar foto')),
      body: FutureBuilder<void>(
        future: _initFuture,
        builder: (context, snapshot) {
          if (snapshot.connectionState != ConnectionState.done) {
            return const Center(child: CircularProgressIndicator());
          }
          return Stack(
            fit: StackFit.expand,
            children: [
              CameraPreview(_controller!),
              Positioned(
                bottom: 24,
                left: 0,
                right: 0,
                child: Center(
                  child: FloatingActionButton.large(
                    onPressed: _capturing ? null : _takePhoto,
                    child: _capturing
                        ? const CircularProgressIndicator(color: Colors.white)
                        : const Icon(Icons.camera_alt),
                  ),
                ),
              ),
            ],
          );
        },
      ),
    );
  }
}
