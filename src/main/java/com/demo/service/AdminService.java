package com.demo.service;

import org.springframework.stereotype.Service;
import org.springframework.beans.factory.annotation.Autowired;
import com.demo.entity.Admin;
import com.demo.repo.AdminRepository;

@Service
public class AdminService {

    @Autowired
    private AdminRepository repo;

    private Admin currentAdmin = null;

    public boolean defaultAdminExists() {
        return repo.existsById("GEID-IND-EDU-2025-000123-7");
    }

    public void createDefaultAdmin() {
        if (!defaultAdminExists()) {
            repo.save(new Admin(
                "GEID-IND-EDU-2025-000123-7",
                "admin@123"
            ));
        }
    }

    public Admin login(String adminId, String password) {
        Admin admin = repo.findByAdminIdAndPassword(adminId, password);
        if (admin != null) currentAdmin = admin;
        return admin;
    }

    public Admin getCurrentAdmin() {
        return currentAdmin;
    }

    public void logout() {
        currentAdmin = null;
    }
}
