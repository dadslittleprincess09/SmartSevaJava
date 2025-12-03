package com.demo.repo;

import org.springframework.data.jpa.repository.JpaRepository;
import com.demo.entity.Admin;

public interface AdminRepository extends JpaRepository<Admin, String> {
    Admin findByAdminIdAndPassword(String adminId, String password);
}
