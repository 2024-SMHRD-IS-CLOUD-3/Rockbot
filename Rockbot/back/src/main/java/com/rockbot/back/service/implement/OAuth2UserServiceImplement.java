package com.rockbot.back.service.implement;

import java.util.Map;

import org.springframework.security.oauth2.client.userinfo.DefaultOAuth2UserService;
import org.springframework.security.oauth2.client.userinfo.OAuth2UserRequest;
import org.springframework.security.oauth2.core.OAuth2AuthenticationException;
import org.springframework.security.oauth2.core.user.OAuth2User;
import org.springframework.stereotype.Service;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.rockbot.back.entity.CustomOAuth2User;
import com.rockbot.back.entity.UserEntity;
import com.rockbot.back.repository.UserRepository;

import lombok.RequiredArgsConstructor;

@Service
@RequiredArgsConstructor
public class OAuth2UserServiceImplement extends DefaultOAuth2UserService {

    private final UserRepository userRepository;

    @Override
    public OAuth2User loadUser(OAuth2UserRequest request) throws OAuth2AuthenticationException {
        OAuth2User oAuth2User = super.loadUser(request);
        String oauthClientName = request.getClientRegistration().getClientName();

        /*
         * try {
         * System.out.println(new
         * ObjectMapper().writeValueAsString(oAuth2User.getAttributes()));
         * } catch (Exception exception) {
         * exception.printStackTrace();
         * }
         */

        UserEntity userEntity = null;
        String userId = null;
        String email = "email@email.com";
        String name = "default_name"; // 기본값 설정

        if (oauthClientName.equals("kakao")) {
            userId = "kakao_" + oAuth2User.getAttributes().get("id");
            name = (String) ((Map<String, Object>) oAuth2User.getAttributes().get("properties")).get("nickname");
            userEntity = new UserEntity(userId, email, "kakao");
        }

        if (oauthClientName.equals("naver")) {
            Map<String, String> responseMap = (Map<String, String>) oAuth2User.getAttributes().get("response");
            userId = "naver_" + responseMap.get("id").substring(0, 14);
            email = responseMap.get("email");
            name = responseMap.get("name");
            userEntity = new UserEntity(userId, email, name, "naver");
        }
        userRepository.save(userEntity);

        return new CustomOAuth2User(userId, name);
    }
}
