package com.demo;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.autoconfigure.domain.EntityScan;
import org.springframework.data.jpa.repository.config.EnableJpaRepositories;
import org.springframework.context.annotation.ComponentScan;

@SpringBootApplication
@ComponentScan(basePackages = "com.demo")   // ⭐ Important
@EnableJpaRepositories(basePackages = {"com.demo.dao", "com.demo.repo"})
@EntityScan(basePackages = {"com.demo.model","com.demo.entity"})
public class SpringBootNotification1Application {

    public static void main(String[] args) {
        SpringApplication.run(SpringBootNotification1Application.class, args);
    }

}
