package com.rockbot.back.handler;

import java.io.IOException;
import java.net.URLEncoder;

import org.springframework.security.core.Authentication;
import org.springframework.security.web.authentication.SimpleUrlAuthenticationSuccessHandler;
import org.springframework.stereotype.Component;

import com.rockbot.back.entity.CustomOAuth2User;
import com.rockbot.back.provider.JwtProvider;

import jakarta.servlet.ServletException;
import jakarta.servlet.http.Cookie;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import lombok.RequiredArgsConstructor;

@Component
@RequiredArgsConstructor
public class OAuth2SuccessHandler extends SimpleUrlAuthenticationSuccessHandler {

    private final JwtProvider jwtProvider;

    @Override
    public void onAuthenticationSuccess(
            HttpServletRequest request,
            HttpServletResponse response,
            Authentication authentication) throws IOException, ServletException {

        CustomOAuth2User oAuth2User = (CustomOAuth2User) authentication.getPrincipal();
        String userId = oAuth2User.getUserId();
        String name = oAuth2User.getName();
        int departmentId = 1;
        String token = jwtProvider.create(userId, name, departmentId);
        int expirationTime = 3600; // 토큰 유효 시간 (초)

        // 쿠키로 정보 설정
        Cookie tokenCookie = new Cookie("accessToken", URLEncoder.encode(token, "UTF-8"));
        tokenCookie.setMaxAge(expirationTime);
        tokenCookie.setPath("/"); // 모든 경로에서 접근 가능
        response.addCookie(tokenCookie);

        Cookie nameCookie = new Cookie("name", URLEncoder.encode(name, "UTF-8"));
        nameCookie.setMaxAge(expirationTime);
        nameCookie.setPath("/");
        response.addCookie(nameCookie);

        Cookie userIdCookie = new Cookie("userId", URLEncoder.encode(userId, "UTF-8"));
        userIdCookie.setMaxAge(expirationTime);
        userIdCookie.setPath("/");
        response.addCookie(userIdCookie);

        // 프론트엔드로 리다이렉트
        response.sendRedirect("http://localhost:3000/auth/chat");
    }
}
