package com.demo.security;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import org.springframework.security.config.annotation.web.configuration.EnableWebSecurity;
import org.springframework.security.config.annotation.web.builders.HttpSecurity;

import org.springframework.security.oauth2.client.userinfo.DefaultOAuth2UserService;
import org.springframework.security.oauth2.client.userinfo.OAuth2UserRequest;

import org.springframework.security.oauth2.core.user.OAuth2User;
import org.springframework.security.web.SecurityFilterChain;

import com.demo.entity.GoogleUser;
import com.demo.repo.GoogleUserRepository;
import com.demo.service.EmailService;

@Configuration
@EnableWebSecurity
public class SecurityConfig {

    @Autowired
    private GoogleUserRepository googleRepo;
    @Autowired
    EmailService emailService;
    @Bean
    public SecurityFilterChain securityFilterChain(HttpSecurity http) throws Exception {

        http
            .csrf(csrf -> csrf.disable()) // Needed for Postman POST requests
            .cors(cors -> {})
            .authorizeHttpRequests(auth -> auth
                .requestMatchers("/api/user/**").permitAll()  // ⭐ Allow Register/Login
                .requestMatchers("/api/admin/**").permitAll()
                .requestMatchers("/api/predict").permitAll()
                .requestMatchers("/api/complaints/**").permitAll()
                .requestMatchers("/", "/error").permitAll()   // allow home page) 
                .anyRequest().authenticated()                 // protect everything else
            )

            .oauth2Login(oauth -> 
                oauth.userInfoEndpoint(userInfo -> 
                    userInfo.userService(this::saveUserAndNotify)
                )
            )
            
            .logout(logout -> 
            logout.logoutSuccessUrl("/")
        );

        return http.build();
    }
    private OAuth2User saveUserAndNotify(OAuth2UserRequest request) {

        DefaultOAuth2UserService service = new DefaultOAuth2UserService();
        OAuth2User user = service.loadUser(request);

        String email = user.getAttribute("email");
        String name = user.getAttribute("name");
        String picture = user.getAttribute("picture");

        // ✔ Save to DB
        googleRepo.save(new GoogleUser(email, name, picture));

        // ✔ Send Email Notification
        emailService.sendEmail(
            email,
            "Login Successfully",
            "Hi " + name + ", you are login to SmartSeva successfully!"
        );

        System.out.println("📧 Google Login Email Sent To: " + email);

        return user;
}}
