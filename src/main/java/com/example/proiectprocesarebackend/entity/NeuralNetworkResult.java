package com.example.proiectprocesarebackend.entity;

import lombok.AllArgsConstructor;
import lombok.Data;
import org.opencv.core.Rect2d;

import java.util.List;

@Data
@AllArgsConstructor
public class NeuralNetworkResult {

  private List<Rect2d> boxes;

  private List<Float> confidences;
}
