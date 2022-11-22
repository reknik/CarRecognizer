package com.example.proiectprocesarebackend;

import org.opencv.core.Core;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.builder.SpringApplicationBuilder;
import org.springframework.context.ConfigurableApplicationContext;
import springfox.documentation.swagger2.annotations.EnableSwagger2;

@SuppressWarnings("checkstyle:MissingJavadocType")
@SpringBootApplication
@EnableSwagger2
public class ProiectProcesareBackendApplication {

  static {
    System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
  }

  public static void main(String[] args) {
    SpringApplicationBuilder builder = new SpringApplicationBuilder(ProiectProcesareBackendApplication.class);

    builder.headless(false);

    ConfigurableApplicationContext context = builder.run(args);
  }

}
