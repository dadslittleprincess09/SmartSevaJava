package com.demo.controller;

import java.util.Map;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import com.demo.dto.LoginRequest;
import com.demo.model.AppUser;
import com.demo.service.EmailService;
import com.demo.service.UserService;

import jakarta.servlet.http.HttpSession;

@RestController
@RequestMapping("/api/user")
//@CrossOrigin("*")
// @CrossOrigin(origins = "http://localhost:5501","https://smart-final.vercel.app/", allowCredentials = "true")
@CrossOrigin(
    origins = {
        "http://localhost:5501",
        "http://127.0.0.1:5501",
        "https://smart-final.vercel.app"
    },
    allowCredentials = "true"
)
public class UserController {

    @Autowired
    UserService service;

    @Autowired
    EmailService emailService;

    // ---------------- REGISTER -----------------
    @PostMapping("/register")
    public ResponseEntity<?> register(@RequestBody AppUser u) {

        if (service.exists(u.getEmail())) {
            return ResponseEntity.ok(Map.of("status", "EXISTS"));
        }

        service.save(u);

        return ResponseEntity.ok(Map.of("status", "REGISTERED"));
    }

    // ---------------- LOGIN (email + password) -----------------
    @PostMapping("/login")
    public ResponseEntity<?> login(@RequestBody LoginRequest req, HttpSession session) {

        AppUser user = service.login(req.getEmail(), req.getPassword());

        if (user != null) {

            // save in session
            session.setAttribute("user", user);

            // optional email
            emailService.sendEmail(
                    req.getEmail(),
                    "Congrats...! Google OfferLetter",
                    "Hi " + user.getName() + ", you are selected for google your ctc is one rupee"
            );

            // unified response
            return ResponseEntity.ok(Map.of(
                    "status", "success",
                    "user", Map.of(
                            "id", user.getId(),
                            "name", user.getName(),
                            "email", user.getEmail()
                    )
            ));
        }

        return ResponseEntity.status(401).body(Map.of("status", "failed"));
    }

    // ---------------- GOOGLE LOGIN / SIGNUP -----------------
    @PostMapping("/google-auth")
    public ResponseEntity<?> googleAuth(@RequestBody Map<String, String> body, HttpSession session) {

        String email = body.get("email");
        String name = body.get("name");

        AppUser existing = service.getByEmail(email);

        AppUser user;
        if (existing != null) {
            user = existing;
        } else {
            user = new AppUser();
            user.setName(name);
            user.setEmail(email);
            user.setPassword("google"); // dummy password
            user = service.save(user);
        }

        session.setAttribute("user", user);

        return ResponseEntity.ok(Map.of(
                "status", "success",
                "user", Map.of(
                        "id", user.getId(),
                        "name", user.getName(),
                        "email", user.getEmail()
                )
        ));
    }

    // ---------------- CURRENT SESSION USER -----------------
    @GetMapping("/current")
    public ResponseEntity<?> currentUser(HttpSession session) {
        AppUser user = (AppUser) session.getAttribute("user");

        if (user == null) {
            return ResponseEntity.status(401).body(Map.of("status", "anonymous"));
        }

        return ResponseEntity.ok(Map.of(
                "status", "ok",
                "user", Map.of(
                        "id", user.getId(),
                        "name", user.getName(),
                        "email", user.getEmail()
                )
        ));
    }

    // ---------------- LOGOUT -----------------
    @PostMapping("/logout")
    public ResponseEntity<?> logout(HttpSession session) {
        session.invalidate();
        return ResponseEntity.ok(Map.of("status", "logged_out"));
    }
}