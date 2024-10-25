package com.rockbot.back.controller;

import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;



@RestController
@RequestMapping("/api/v1/user")
public class UserContoller {
    
    @GetMapping("/aa")
    public String aa() {
        return "aa";
    }

}
    

