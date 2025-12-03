package com.demo.service;

import com.demo.entity.Complaint;
import com.demo.repo.ComplaintRepository;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.Optional;

@Service
public class ComplaintService {

    private final ComplaintRepository repo;

    public ComplaintService(ComplaintRepository repo) {
        this.repo = repo;
    }

    public Complaint save(Complaint c) {
        return repo.save(c);
    }

    public List<Complaint> getAll() {
        return repo.findAll();
    }

    public List<Complaint> getByUserId(Long userId) {
        return repo.findByUserId(userId);
    }
    public Optional<Complaint> getById(Long id) {
        return repo.findById(id);
    }
}
