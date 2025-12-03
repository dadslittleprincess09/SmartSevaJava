package com.demo.repo;

import org.springframework.data.jpa.repository.JpaRepository;
import com.demo.entity.GoogleUser;

public interface GoogleUserRepository extends JpaRepository<GoogleUser, String> {}
