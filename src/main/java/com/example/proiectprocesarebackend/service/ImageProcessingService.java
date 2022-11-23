package com.example.proiectprocesarebackend.service;


import com.example.proiectprocesarebackend.entity.NeuralNetworkResult;
import net.sourceforge.tess4j.Tesseract;
import net.sourceforge.tess4j.TesseractException;
import org.opencv.core.*;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.utils.Converters;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.util.CollectionUtils;
import org.springframework.web.multipart.MultipartFile;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

@Service
public class ImageProcessingService {
  @Value("${ar.min}")
  private double minAspectRatio;
  @Value("${ar.max}")
  private double maxAspectRatio;
  @Value("${myLicensePlateText}")
  private String myLicensePlateText;

  @Value("${yoloV3.cfg.path}")
  private String yoloV3CfgPath;

  @Value("${yoloV3.weights.path}")
  private String yoloV3WeightsPath;

  private boolean carIsMatchingColor(Mat imageMatrix, List<Rect2d> boxes, MatOfInt indices) {
    List<Integer> indicesList = indices.toList();

    for (int i = 0; i < boxes.size(); i++) {
      //se iau doar elementele determinate ca fiind mai de incredere
      if (indicesList.contains(i)) {
        Rect2d box = boxes.get(i);
        //se taie portiunea din imagine asociata elementului
        Rect roi = new Rect((int) box.x, (int) box.y, (int) box.height, (int) box.width);
        Mat car = imageMatrix.submat(roi);
        Mat inrange = new Mat();
        int colorPixels = 0;
        //se verifica valorile pixelilor
        Core.inRange(car, new Scalar(87, 87, 111), new Scalar(180, 255, 255), inrange);
        colorPixels += Core.countNonZero(inrange);
        if (colorPixels > (inrange.total() / 4)) {
          return true;
        }
      }
    }
    return false;
  }

  /**
   * Method that checks image based on license plate text, classification of car and color.
   *
   * @param imageToProcess MultipartFile, file of a valid image
   * @return whether image is of predefined car
   */
  public boolean processImage(MultipartFile imageToProcess) {
    BufferedImage image;
    Mat imageMatrix;
    try {
      image = ImageIO.read(imageToProcess.getInputStream());
      imageMatrix = Imgcodecs.imdecode(new MatOfByte(imageToProcess.getBytes()), Imgcodecs.IMREAD_UNCHANGED);
    } catch (IOException e) {
      throw new IllegalArgumentException("Couldn't process image", e);
    }

    imageMatrix.put(0, 0, ((DataBufferByte) image.getRaster().getDataBuffer()).getData());
    String text = getLicensePlateText(imageMatrix);

    if (text == null || !text.contains(myLicensePlateText)) {
      return false;
    }

    Net dnnNet = getNeuralNetwork();
    NeuralNetworkResult result = forwardImageOverNetwork(imageMatrix, dnnNet);

    if (result.getBoxes().isEmpty()) {
      return false;
    }

    List<Rect2d> boxes = result.getBoxes();
    List<Float> confidences = result.getConfidences();

    //Non-maximum supression
    //Se aleg indicii elementelor ce au cea mai mare incredere si care nu se intersecteaza
    MatOfInt indices = getBBoxIndicesFromNonMaximumSuppression(boxes,
        confidences);

    return carIsMatchingColor(imageMatrix, boxes, indices);
  }

  private Net getNeuralNetwork() {
    Net dnnNet = Dnn.readNetFromDarknet(yoloV3CfgPath, yoloV3WeightsPath);
    //Procesarea se face mai rapid pe placa video din cauza paralelizarii mai bune cu ajutorul CUDA
    //    dnnNet.setPreferableBackend(Dnn.DNN_BACKEND_CUDA);
    //    dnnNet.setPreferableTarget(Dnn.DNN_TARGET_CUDA);
    return dnnNet;
  }

  private NeuralNetworkResult forwardImageOverNetwork(Mat img, Net dnnNet) {

    List<Rect2d> boxes = new ArrayList<>();
    List<Float> confidences = new ArrayList<>();
    List<String> layerNames = dnnNet.getLayerNames();
    List<String> outputLayers = new ArrayList<>();
    for (Integer i : dnnNet.getUnconnectedOutLayers().toList()) {
      outputLayers.add(layerNames.get(i - 1));
    }

    //Se creaza matricea de pixeli ce poate fi citita de reteaua neuronala
    Mat neuralNetworkImage = Dnn.blobFromImage(img, 1 / 255.0, new Size(416, 416),
        new Scalar(new double[]{0.0, 0.0, 0.0}), true, false);

    dnnNet.setInput(neuralNetworkImage);

    List<Mat> outputs = new ArrayList<>();

    dnnNet.forward(outputs, outputLayers);

    for (Mat output : outputs) {
      for (int i = 0; i < output.rows(); i++) {
        Mat row = output.row(i);
        List<Float> detect = new MatOfFloat(row).toList();
        List<Float> score = detect.subList(5, output.cols());
        int classId = getIndicesOfMaximum(score);
        float conf = score.get(classId);
        if (conf >= 0.9 && classId == 2) {
          int centerX = (int) (detect.get(0) * img.cols());
          int centerY = (int) (detect.get(1) * img.rows());
          int width = (int) (detect.get(2) * img.cols());
          int height = (int) (detect.get(3) * img.rows());
          int x = (centerX - width / 2);
          int y = (centerY - height / 2);
          Rect2d box = new Rect2d(x, y, width, height);
          boxes.add(box);
          confidences.add(conf);
        }
      }
    }
    return new NeuralNetworkResult(boxes, confidences);
  }

  private int getIndicesOfMaximum(List<Float> array) {
    int maxIndices = 0;

    for (int i = 0; i < array.size(); i++) {
      maxIndices = array.get(i) > array.get(maxIndices) ? i : maxIndices;
    }

    return maxIndices;
  }

  private MatOfInt getBBoxIndicesFromNonMaximumSuppression(List<Rect2d> boxes, List<Float> confidences) {
    MatOfRect2d matOfBoxes = new MatOfRect2d();
    matOfBoxes.fromList(boxes);
    MatOfFloat matOfConfidences = new MatOfFloat(Converters.vector_float_to_Mat(confidences));
    MatOfInt result = new MatOfInt();
    Dnn.NMSBoxes(matOfBoxes, matOfConfidences, (float) (0.6), (float) (0.5), result);
    return result;
  }

  private String getLicensePlateText(Mat imageMatrix) {
    Mat grayScale = new Mat();
    imageMatrix.copyTo(grayScale);
    Imgproc.cvtColor(grayScale, grayScale, Imgproc.COLOR_RGB2GRAY);
    List<MatOfPoint> contours = locateLicenseContour(grayScale);

    List<Mat> licensePlateCandidates = locateLicensePlate(grayScale, contours);

    if (CollectionUtils.isEmpty(licensePlateCandidates)) {
      return null;
    }

    StringBuilder text = new StringBuilder("");

    Tesseract tesseract = new Tesseract();
    for (Mat licensePlateCandidate : licensePlateCandidates) {
      String licensePlateText;
      try {
        licensePlateText = tesseract.doOCR(toBufferedImage(licensePlateCandidate));
        text.append(licensePlateText.replaceAll("[^a-zA-Z0-9]", "").toUpperCase());
      } catch (TesseractException e) {
        throw new RuntimeException(e);
      }
    }
    return text.toString();
  }

  private List<MatOfPoint> locateLicenseContour(Mat grayScale) {
    Mat blackHatRect = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(7, 5));
    Mat blackHat = new Mat();
    //Determinam obiectele intunecate in zone luminate e.g. text-ul placutei
    Imgproc.morphologyEx(grayScale, blackHat, Imgproc.MORPH_BLACKHAT, blackHatRect);

    HighGui.imshow("Blackhat", blackHat);
    HighGui.waitKey();

    //Determinam obiectele intunecate, reducem zgomotul si dupa aplicam o transformare binara folosind metoda lui otsu
    Mat squareRect = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(3, 3));
    Mat light = new Mat();
    Imgproc.morphologyEx(grayScale, light, Imgproc.MORPH_CLOSE, squareRect);
    Imgproc.threshold(light, light, 0, 255, Imgproc.THRESH_BINARY | Imgproc.THRESH_OTSU);
    HighGui.imshow("Light", light);
    HighGui.waitKey();
    Mat gradX = new Mat();

    //Folosind Sobel, se detecteaza conturul obiectelor din zonele intunecate
    Imgproc.Sobel(blackHat, gradX, CvType.CV_32F, 1, 0, -1);
    Core.absdiff(gradX, Scalar.all(0), gradX);
    Core.MinMaxLocResult result = Core.minMaxLoc(gradX);
    Core.subtract(gradX, new Scalar(result.minVal), gradX);
    Core.divide(gradX, new Scalar(result.maxVal - result.minVal), gradX);
    Core.multiply(gradX, new Scalar(255), gradX);
    gradX.convertTo(gradX, CvType.CV_8UC1);
    HighGui.imshow("Sobel", gradX);
    HighGui.waitKey();

    //Se netezeste imaginea si se aplica transformarea binara otsu din nou
    Imgproc.GaussianBlur(gradX, gradX, new Size(41, 13), 0);
    Imgproc.morphologyEx(gradX, gradX, Imgproc.MORPH_CLOSE, blackHatRect);
    Mat threshold = new Mat();
    Imgproc.threshold(gradX, threshold, 0, 255, Imgproc.THRESH_BINARY | Imgproc.THRESH_OTSU);
    HighGui.imshow("Grad Thresh", threshold);
    HighGui.waitKey();

    //Se reduce din zgomot
    Imgproc.erode(threshold, threshold, new Mat(), new Point(), 2);
    Imgproc.dilate(threshold, threshold, new Mat(), new Point(), 2);
    HighGui.imshow("Grad Erode Dilate", threshold);
    HighGui.waitKey();

    //Se suprapun zonele luminate
    Core.bitwise_and(threshold, threshold, light);
    Imgproc.dilate(threshold, threshold, new Mat(), new Point(), 2);
    Imgproc.erode(threshold, threshold, new Mat(), new Point(), 1);
    HighGui.imshow("Final", threshold);
    HighGui.waitKey();
    List<MatOfPoint> contours = new ArrayList<>();
    Mat copy = new Mat();
    threshold.copyTo(copy);

    //se cauta contururile si se sorteaza
    Imgproc.findContours(copy, contours, new Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
    contours.sort(Comparator.comparing(contour -> Imgproc.contourArea((Mat) contour), Double::compareTo).reversed());
    return contours;
  }

  private List<Mat> locateLicensePlate(Mat grayScale, List<MatOfPoint> contours) {

    List<Mat> licensePlateCandidates = new ArrayList<>();
    for (MatOfPoint contour : contours) {
      Rect contourRect = Imgproc.boundingRect(contour);
      double aspectRatio = contourRect.width / (double) contourRect.height;
      if (aspectRatio >= minAspectRatio && aspectRatio <= maxAspectRatio) {
        Mat licensePlate = grayScale.submat(contourRect.y,
            contourRect.y + contourRect.height, contourRect.x, contourRect.x + contourRect.width);
        Mat regionOfInterest = new Mat();
        Imgproc.threshold(licensePlate, regionOfInterest, 0, 255, Imgproc.THRESH_BINARY_INV | Imgproc.THRESH_OTSU);
        licensePlateCandidates.add(regionOfInterest);
      }
    }
    return licensePlateCandidates;
  }

  private BufferedImage toBufferedImage(Mat m) {
    if (!m.empty()) {
      int type = BufferedImage.TYPE_BYTE_GRAY;
      if (m.channels() > 1) {
        type = BufferedImage.TYPE_3BYTE_BGR;
      }
      int bufferSize = m.channels() * m.cols() * m.rows();
      byte[] b = new byte[bufferSize];
      m.get(0, 0, b); // get all the pixels
      BufferedImage image = new BufferedImage(m.cols(), m.rows(), type);
      final byte[] targetPixels = ((DataBufferByte) image.getRaster().getDataBuffer()).getData();
      System.arraycopy(b, 0, targetPixels, 0, b.length);
      return image;
    }

    return null;
  }

  private void showImage(Mat image) {
    HighGui.imshow("img", image);
    HighGui.waitKey(1500);
  }

  private void showImage(Mat image, String name) {
    HighGui.imshow(name, image);
    HighGui.waitKey();
  }
}
