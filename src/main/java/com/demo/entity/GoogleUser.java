package com.demo.entity;

import jakarta.persistence.Entity;
import jakarta.persistence.Id;
import jakarta.persistence.Table;

@Entity
@Table(name = "google_users")  // Better to avoid clash with other "users" table
public class GoogleUser {

    @Id
    private String email;   // Primary Key

    private String name;
    private String picture;

    public GoogleUser() { 
        // Required by JPA
    }

    public GoogleUser(String email, String name, String picture) {
        this.email = email;
        this.name = name;
        this.picture = picture;
    }

    // Getters & Setters
    public String getEmail() {
        return email;
    }
    public void setEmail(String email) {
        this.email = email;
    }

    public String getName() {
        return name;
    }
    public void setName(String name) {
        this.name = name;
    }

    public String getPicture() {
        return picture;
    }
    public void setPicture(String picture) {
        this.picture = picture;
    }
}
