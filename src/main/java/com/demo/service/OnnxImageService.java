package com.demo.service;

import java.awt.Graphics2D;
import java.awt.Image;
import java.awt.image.BufferedImage;
import java.io.File;
import java.net.URL;
import java.nio.FloatBuffer;
import java.util.Arrays;
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

        // Load all 4 models  
        categoryModel = loadModel("/model/main_category_model.onnx");
        childModel = loadModel("/model/child_severity_model.onnx");
        roadModel = loadModel("/model/road_severity_model.onnx");
        garbageModel = loadModel("/model/garbage_severity_model.onnx");
    }

    // Utility to load a model from resource folder
    private OrtSession loadModel(String path) throws Exception {
        URL url = getClass().getResource(path);
        if (url == null) {
            throw new RuntimeException("Model file not found: " + path);
        }
        File modelFile = new File(url.toURI());
        return env.createSession(modelFile.getAbsolutePath(), new OrtSession.SessionOptions());
    }

    // Convert image to [1,224,224,3]
    // Convert image to [1,224,224,3] WITHOUT dividing by 255
private float[] preprocessImage(File file) throws Exception {
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

            // Extract RGB (0–255)
            float r = (pixel >> 16) & 0xFF;
            float gVal = (pixel >> 8) & 0xFF;
            float b = pixel & 0xFF;

            if (x == 0 && y == 0) {
                System.out.println("FIRST PIXEL RAW R,G,B = " + r + ", " + gVal + ", " + b);
            }

            // DO NOT DIVIDE BY 255
            input[idx++] = r;
            input[idx++] = gVal;
            input[idx++] = b;
        }
    }

    return input;
}


    // Main category prediction (Model 1)
    private String predictCategory(float[] inputData) throws Exception {
        long[] shape = new long[]{1, 224, 224, 3};
        OnnxTensor inputTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(inputData), shape);

        Map<String, OnnxTensor> input = new HashMap<>();
        input.put("input", inputTensor);

        OrtSession.Result result = categoryModel.run(input);

        float[] out = ((float[][]) result.get(0).getValue())[0];
        System.out.println("CATEGORY RAW OUTPUT = " + Arrays.toString(out));

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

    // Severity Model for all categories
    private String runSeverityModel(OrtSession model, float[] inputData) throws Exception {
        long[] shape = new long[]{1, 224, 224, 3};
        OnnxTensor inputTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(inputData), shape);

        Map<String, OnnxTensor> input = new HashMap<>();
        input.put("input", inputTensor);

        OrtSession.Result result = model.run(input);

        float[] out = ((float[][]) result.get(0).getValue())[0];

System.out.println("Category model output: " + Arrays.toString(out));
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

    // Final pipeline → category model → severity model
    public Map<String, String> predictPipeline(File file) throws Exception {

        float[] inputData = preprocessImage(file);

        String category = predictCategory(inputData);
        String severity = "";

        switch(category) {
            case "Child":
                severity = runSeverityModel(childModel, inputData);
                break;
            case "Road":
                severity = runSeverityModel(roadModel, inputData);
                break;
            case "Garbage":
                severity = runSeverityModel(garbageModel, inputData);
                break;
        }

        Map<String, String> out = new HashMap<>();
        out.put("category", category);
        out.put("severity", severity);

        return out;
    }
}