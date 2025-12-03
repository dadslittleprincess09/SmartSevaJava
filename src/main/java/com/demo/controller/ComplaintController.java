package com.demo.controller;

import com.demo.dao.AppUserRepository;
import com.demo.entity.Complaint;
import com.demo.model.AppUser;
import com.demo.repo.ComplaintRepository;
import com.demo.service.ComplaintService;
import com.demo.service.EmailService;

import jakarta.servlet.http.HttpSession;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.io.File;
import java.util.Map;

@RestController
@RequestMapping("/api/complaints")
@CrossOrigin(
        origins = {"http://127.0.0.1:5501", "http://localhost:5501"},
        allowCredentials = "true",
        methods = {RequestMethod.GET, RequestMethod.POST, RequestMethod.PUT, RequestMethod.DELETE, RequestMethod.PATCH, RequestMethod.OPTIONS}
)
public class ComplaintController {

    @Value("${smartseva.admin.email}")
    private String adminEmail;

    @Autowired
    private AppUserRepository appUserRepository;

    @Autowired
    private ComplaintRepository complaintRepository;

    @Autowired
    private EmailService emailService;

    private final ComplaintService complaintService;

    public ComplaintController(ComplaintService complaintService) {
        this.complaintService = complaintService;
    }

    // ============================================
    //   CREATE COMPLAINT  (/add)
    // ============================================
    @PostMapping("/add")
    public ResponseEntity<?> addComplaint(
            @RequestPart("category") String category,
            @RequestPart("description") String description,
            @RequestPart("location") String location,
            @RequestPart("severity") String severity,
            @RequestPart(value = "image", required = false) MultipartFile image,
            HttpSession session
    ) {
        try {
            // Check user login
            AppUser user = (AppUser) session.getAttribute("user");
            if (user == null) {
                return ResponseEntity.status(401).body(Map.of("status", "NOT_LOGGED_IN"));
            }

            // Reject LOW severity
            if (severity != null && severity.equalsIgnoreCase("LOW")) {
                return ResponseEntity.status(400).body(
                        Map.of("status", "LOW_SEVERITY_REJECTED")
                );
            }

            // Create complaint
            Complaint c = new Complaint();
            c.setUserId(user.getId());
            c.setUserName(user.getName());
            c.setCategory(category);
            c.setDescription(description != null ? description : "");
            c.setLocation(location);
            c.setSeverity(severity);
            c.setStatus("PENDING");

            // Save image
         // Save image
            if (image != null && !image.isEmpty()) {

                String fileName = System.currentTimeMillis() + "_" + image.getOriginalFilename();

                String uploadDir = System.getProperty("user.dir") + "/uploads/";

                File uploadPath = new File(uploadDir);
                if (!uploadPath.exists()) uploadPath.mkdirs();

                File dest = new File(uploadDir + fileName);
                image.transferTo(dest);

                c.setImageUrl(fileName);
            }


            // Save to DB
            complaintService.save(c);

            // Send alert mail to admin
            emailService.sendNewComplaintAlertToAdmin(adminEmail, c);

            return ResponseEntity.ok(Map.of("status", "COMPLAINT_SAVED"));

        } catch (Exception e) {
            e.printStackTrace();
            return ResponseEntity.status(500).body("Error: " + e.getMessage());
        }
    }

    // ============================================
    //   UPDATE STATUS (Admin)
    // ============================================
    @PatchMapping("/update-status/{id}")
    public ResponseEntity<?> updateStatus(
            @PathVariable Long id,
            @RequestBody Map<String, String> body
    ) {
        String newStatus = body.get("status");
        if (newStatus == null) {
            return ResponseEntity.badRequest().body(Map.of("status", "INVALID_STATUS"));
        }

        Complaint complaint = complaintService.getById(id)
                .orElseThrow(() -> new RuntimeException("Complaint not found"));

        complaint.setStatus(newStatus.toUpperCase());
        complaintService.save(complaint);

        return ResponseEntity.ok(Map.of("status", "STATUS_UPDATED"));
    }

    // ============================================
    //   GET MY COMPLAINTS (USER)
    // ============================================
    @GetMapping("/my")
    public ResponseEntity<?> myComplaints(HttpSession session) {
        AppUser user = (AppUser) session.getAttribute("user");

        if (user == null) {
            return ResponseEntity.status(401).body(Map.of("status", "NOT_LOGGED_IN"));
        }

        return ResponseEntity.ok(complaintService.getByUserId(user.getId()));
    }

    // ============================================
    //   GET ALL COMPLAINTS (ADMIN)
    // ============================================
    @GetMapping("/all")
    public ResponseEntity<?> allComplaints() {
        return ResponseEntity.ok(complaintService.getAll());
    }

    // ============================================
    //   SEND EMAIL TO USER (When admin resolves)
    // ============================================
    @PostMapping("/send-email/{id}")
    public ResponseEntity<?> sendEmailToUser(@PathVariable Long id) {
        try {
            Complaint c = complaintRepository.findById(id)
                    .orElseThrow(() -> new RuntimeException("Complaint not found"));

            AppUser user = appUserRepository.findById(c.getUserId())
                    .orElseThrow(() -> new RuntimeException("User not found"));

            emailService.sendComplaintResolvedEmail(
                    user.getEmail(),
                    c.getUserName(),
                    String.valueOf(c.getUserId())
            );

            return ResponseEntity.ok("Email sent successfully!");

        } catch (Exception e) {
            return ResponseEntity.status(500).body("Failed to send email: " + e.getMessage());
        }
    }
}
