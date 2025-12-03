package com.demo.controller;

import org.springframework.web.bind.annotation.*;
import org.springframework.beans.factory.annotation.Autowired;

import com.demo.entity.Admin;
import com.demo.service.AdminService;

@RestController
@RequestMapping("/api/admin")
@CrossOrigin("*")
public class AdminController {

    @Autowired
    AdminService adminService;

    // 🔹 Initialize default admin once
    @GetMapping("/init")
    public String initializeAdmin() {
        adminService.createDefaultAdmin();
        return "DEFAULT_ADMIN_CREATED";
    }

    @PostMapping("/login")
    public Object adminLogin(@RequestBody Admin adm) {
        Admin a = adminService.login(adm.getAdminId(), adm.getPassword());
        if (a == null) return "INVALID_ADMIN";
        return a;
    }

    @GetMapping("/current")
    public Admin getCurrentAdmin() {
        return adminService.getCurrentAdmin();
    }

    @PostMapping("/logout")
    public String adminLogout() {
        adminService.logout();
        return "ADMIN_LOGGED_OUT";
    }
}
