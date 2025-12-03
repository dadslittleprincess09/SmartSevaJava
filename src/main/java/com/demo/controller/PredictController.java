package com.demo.controller;

import java.io.File;
import java.util.Map;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

import com.demo.service.OnnxImageService;

@CrossOrigin(origins = "*")
@RestController
@RequestMapping("/api")
public class PredictController {

    @Autowired
    private OnnxImageService onnxImageService;

    @PostMapping("/predict")
    public ResponseEntity<?> predict(@RequestParam("file") MultipartFile file) {

        try {
            if (file.isEmpty()) {
                return ResponseEntity.badRequest().body("File is empty!");
            }

            // Save file temporarily
            File conv = File.createTempFile("upload_", "_" + file.getOriginalFilename());
            file.transferTo(conv);

            // NEW PIPELINE CALL (category → severity)
            Map<String, String> result = onnxImageService.predictPipeline(conv);

            // Delete temp file
            conv.delete();

            // Return result to frontend
            return ResponseEntity.ok(Map.of(
                "category", result.get("category"),
                "severity", result.get("severity")
            ));

        } catch (Exception e) {
            e.printStackTrace();
            return ResponseEntity.status(500).body("Error: " + e.getMessage());
        }
    }
}