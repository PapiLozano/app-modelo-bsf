import 'package:flutter/material.dart';
import 'home_screen.dart';

void main() {
  runApp(const BsfApp());
}

class BsfApp extends StatelessWidget {
  const BsfApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Conteo BSF',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(colorSchemeSeed: Colors.teal, useMaterial3: true),
      home: const HomeScreen(),
    );
  }
}
