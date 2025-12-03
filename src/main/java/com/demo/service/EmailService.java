package com.demo.service;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.mail.SimpleMailMessage;
import org.springframework.mail.javamail.JavaMailSender;
import org.springframework.stereotype.Service;

import com.demo.entity.Complaint;

@Service
public class EmailService {

    @Autowired
    private JavaMailSender mailSender;

    public void sendEmail(String to, String subject, String message) {
    	System.out.println("------ EMAIL SENDING START -------");
        System.out.println("To      : " + to);
        System.out.println("Subject : " + subject);
        System.out.println("Message : " + message);
        
        try {
        SimpleMailMessage mail = new SimpleMailMessage();
        mail.setFrom("smartsevaforpeople@gmail.com"); 
        mail.setTo(to);
        mail.setSubject(subject);
        mail.setText(message);
        mailSender.send(mail);
        System.out.println("------ EMAIL SENT SUCCESSFULLY ------");
        }catch(Exception e) {
        	System.out.println("------ EMAIL FAILED ------");
            e.printStackTrace();
        }
    }
    
public void sendComplaintResolvedEmail(String email, String userName, String tokenId) {
    String subject = "Your Complaint Has Been Resolved";
    String message =
            "Hello " + userName + ",\n\n" +
            "Your complaint with token ID: " + tokenId + " has been RESOLVED.\n\n" +
            "Thank you for using SmartSeva.\n\n" +
            "Regards,\nSmartSeva Team";

    sendEmail(email, subject, message);
}

public void sendNewComplaintAlertToAdmin(String adminEmail, Complaint c) {
    String subject = "New Complaint Submitted - SmartSeva";

    String message =
            "Hello Admin,\n\n" +
            "A new complaint has been submitted by: " + c.getUserName() + "\n\n" +
            "Category: " + c.getCategory() + "\n" +
            "Severity: " + c.getSeverity() + "\n" +
            "Location: " + c.getLocation() + "\n" +
            "Token ID: " + c.getUserId() + "\n\n" +
            "Please review it in the Admin Dashboard.\n\n" +
            "Regards,\nSmartSeva System";

    sendEmail(adminEmail, subject, message);
}


}