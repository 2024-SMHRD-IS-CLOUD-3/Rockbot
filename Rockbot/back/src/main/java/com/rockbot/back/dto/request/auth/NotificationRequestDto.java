package com.rockbot.back.dto.request.auth;

import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

@Getter
@Setter
@NoArgsConstructor
public class NotificationRequestDto {
    private String userId;
    private String message;

    // 생성자
    public NotificationRequestDto(String userId, String message) {
        this.userId = userId;
        this.message = message;
    }
}