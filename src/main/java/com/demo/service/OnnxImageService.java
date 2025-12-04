package com.demo.service;

import java.awt.Graphics2D;
import java.awt.Image;
import java.awt.image.BufferedImage;
import java.io.InputStream;
import java.nio.FloatBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.HashMap;
import java.util.Map;

import javax.imageio.ImageIO;

import org.springframework.stereotype.Service;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;

@Service
public class OnnxImageService {

    private final OrtEnvironment env;
    private OrtSession categoryModel;
    private OrtSession childModel;
    private OrtSession roadModel;
    private OrtSession garbageModel;

    public OnnxImageService() throws Exception {
        env = OrtEnvironment.getEnvironment();

        // Load models from JAR correctly (Render-compatible)
        categoryModel = loadModel("/model/main_category_model.onnx");
        childModel = loadModel("/model/child_severity_model.onnx");
        roadModel = loadModel("/model/road_severity_model.onnx");
        garbageModel = loadModel("/model/garbage_severity_model.onnx");
    }

    // FIXED: Load ONNX model using InputStream → temp file (works in Docker)
    private OrtSession loadModel(String resourcePath) throws Exception {
        InputStream is = getClass().getResourceAsStream(resourcePath);
        if (is == null) {
            throw new RuntimeException("Model not found: " + resourcePath);
        }

        Path temp = Files.createTempFile("model", ".onnx");
        Files.copy(is, temp, StandardCopyOption.REPLACE_EXISTING);

        return env.createSession(temp.toString(), new OrtSession.SessionOptions());
    }

    // Convert image to float array for ONNX
    private float[] preprocessImage(java.io.File file) throws Exception {
        BufferedImage img = ImageIO.read(file);

        Image scaled = img.getScaledInstance(224, 224, Image.SCALE_SMOOTH);
        BufferedImage resized = new BufferedImage(224, 224, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = resized.createGraphics();
        g.drawImage(scaled, 0, 0, null);
        g.dispose();

        float[] input = new float[1 * 224 * 224 * 3];
        int idx = 0;

        for (int y = 0; y < 224; y++) {
            for (int x = 0; x < 224; x++) {
                int pixel = resized.getRGB(x, y);
                input[idx++] = (pixel >> 16) & 0xFF; // R
                input[idx++] = (pixel >> 8) & 0xFF;  // G
                input[idx++] = pixel & 0xFF;         // B
            }
        }

        return input;
    }

    private String predictCategory(float[] inputData) throws Exception {
        long[] shape = new long[]{1, 224, 224, 3};
        OnnxTensor inputTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(inputData), shape);

        Map<String, OnnxTensor> input = new HashMap<>();
        input.put("input", inputTensor);

        OrtSession.Result result = categoryModel.run(input);

        float[] out = ((float[][]) result.get(0).getValue())[0];

        int maxIndex = 0;
        for (int i = 1; i < out.length; i++) {
            if (out[i] > out[maxIndex]) maxIndex = i;
        }

        return switch (maxIndex) {
            case 0 -> "Child";
            case 1 -> "Garbage";
            case 2 -> "Road";
            default -> "Unknown";
        };
    }

    private String runSeverityModel(OrtSession model, float[] inputData) throws Exception {
        long[] shape = new long[]{1, 224, 224, 3};
        OnnxTensor inputTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(inputData), shape);

        Map<String, OnnxTensor> input = new HashMap<>();
        input.put("input", inputTensor);

        OrtSession.Result result = model.run(input);
        float[] out = ((float[][]) result.get(0).getValue())[0];

        int maxIdx = 0;
        for (int i = 1; i < out.length; i++) {
            if (out[i] > out[maxIdx]) maxIdx = i;
        }

        return switch (maxIdx) {
            case 0 -> "Low";
            case 1 -> "High";
            default -> "Unknown";
        };
    }

    public Map<String, String> predictPipeline(java.io.File file) throws Exception {
        float[] inputData = preprocessImage(file);

        String category = predictCategory(inputData);
        String severity = "";

        switch (category) {
            case "Child" -> severity = runSeverityModel(childModel, inputData);
            case "Road" -> severity = runSeverityModel(roadModel, inputData);
            case "Garbage" -> severity = runSeverityModel(garbageModel, inputData);
        }

        Map<String, String> out = new HashMap<>();
        out.put("category", category);
        out.put("severity", severity);

        return out;
    }
}
