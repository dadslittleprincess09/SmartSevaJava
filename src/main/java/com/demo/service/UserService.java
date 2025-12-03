package com.demo.service;

import org.springframework.stereotype.Service;
import org.springframework.beans.factory.annotation.Autowired;
import com.demo.dao.AppUserRepository;
import com.demo.model.AppUser;

@Service
public class UserService {

    @Autowired
    AppUserRepository repo;

    public boolean exists(String email) {
        return repo.findByEmail(email).isPresent();
    }

    public AppUser save(AppUser u) {
        return repo.save(u);
    }

    public AppUser login(String email, String password) {
        return repo.findByEmailAndPassword(email, password).orElse(null);
    }

    public AppUser getByEmail(String email) {
        return repo.findByEmail(email).orElse(null);
    }
}