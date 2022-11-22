package com.example.proiectprocesarebackend.controller;

import com.example.proiectprocesarebackend.service.ImageProcessingService;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.multipart.MultipartFile;

/**
 * Main controller of BE image processing application.
 *
 */
@Controller
public class ImageController {

  private final ImageProcessingService imageProcessingService;


  public ImageController(ImageProcessingService imageProcessingService) {
    this.imageProcessingService = imageProcessingService;
  }

  /**
   * Endpoint for determining whether image is of the predefined car or not.
   *
   * @param imageToProcess MultipartFile, valid image received as multipart
   * @return Boolean, whether predefined car is in image or not
   */
  @PostMapping(value = "/process-image")
  public ResponseEntity<Boolean> processImage(@RequestParam MultipartFile imageToProcess) {
    try {
      return ResponseEntity.ok(imageProcessingService.processImage(imageToProcess));
    } catch (IllegalArgumentException e) {
      ResponseEntity.badRequest();
    }
    return null;
  }
}
