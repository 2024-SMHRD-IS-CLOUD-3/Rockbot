package com.rockbot.back.service.implement;

import org.springframework.http.ResponseEntity;
import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;

import com.rockbot.back.common.CertificationNumber;
import com.rockbot.back.dto.request.auth.CheckCertificationRequestDto;
import com.rockbot.back.dto.request.auth.EmailCertificationRequestDto;
import com.rockbot.back.dto.request.auth.IdCheckRequestDto;
import com.rockbot.back.dto.request.auth.SignInRequestDto;
import com.rockbot.back.dto.request.auth.SignUpRequestDto;
import com.rockbot.back.dto.response.ResponseDto;
import com.rockbot.back.dto.response.auth.CheckCertificationResponseDto;
import com.rockbot.back.dto.response.auth.EmailCertificationResponseDto;
import com.rockbot.back.dto.response.auth.IdCheckResponseDto;
import com.rockbot.back.dto.response.auth.SignInResponseDto;
import com.rockbot.back.dto.response.auth.SignUpResponseDto;
import com.rockbot.back.entity.CertificationEntity;
import com.rockbot.back.entity.UserEntity;
import com.rockbot.back.provider.EmailProvider;
import com.rockbot.back.provider.JwtProvider;
import com.rockbot.back.repository.CertificationRepository;
import com.rockbot.back.repository.UserRepository;
import com.rockbot.back.service.AuthService;

import lombok.RequiredArgsConstructor;

@Service
@RequiredArgsConstructor
public class AuthServiceImplement implements AuthService {

    private final UserRepository userRepository;
    private final CertificationRepository certificationRepository;

    private final EmailProvider emailProvider;
    private final JwtProvider jwtProvider;
    private PasswordEncoder passwordEncoder = new BCryptPasswordEncoder();

    @Override
    public ResponseEntity<? super IdCheckResponseDto> idCheck(IdCheckRequestDto dto) {

        try {

            String userId = dto.getId();
            boolean isExistId = userRepository.existsById(userId);
            if (isExistId)
                return IdCheckResponseDto.duplicateId();

        } catch (Exception exception) {
            exception.printStackTrace();
            return ResponseDto.databaseError();
        }

        return IdCheckResponseDto.success();

    }

    @Override
    public ResponseEntity<? super EmailCertificationResponseDto> emailCertification(EmailCertificationRequestDto dto) {

        try {

            String userId = dto.getId();
            String email = dto.getEmail();

            boolean isExistId = userRepository.existsByUserId(userId);
            if (isExistId)
                return EmailCertificationResponseDto.duplicateID();

            String certificationNumber = CertificationNumber.getCertificationNumber();

            boolean isSuccessed = emailProvider.sendCertificationMail(email, certificationNumber);
            if (!isSuccessed)
                return EmailCertificationResponseDto.mailSendFail();

            CertificationEntity certificationEntity = new CertificationEntity(userId, email, certificationNumber);
            certificationRepository.save(certificationEntity);

        } catch (Exception exception) {
            exception.printStackTrace();
            return ResponseDto.databaseError();
        }

        return EmailCertificationResponseDto.success();
    }

    @Override
    public ResponseEntity<? super CheckCertificationResponseDto> checkCertification(CheckCertificationRequestDto dto) {

        try {

            String userId = dto.getId();
            String email = dto.getEmail();
            String certificationNumber = dto.getCertificationNumber();

            CertificationEntity certificationEntity = certificationRepository.findByUserId(userId);
            if (certificationEntity == null)
                return CheckCertificationResponseDto.certificationFail();

            boolean isMatched = certificationEntity.getEmail().equals(email)
                    && certificationEntity.getCertificationNumber().equals(certificationNumber);
            if (!isMatched)
                return CheckCertificationResponseDto.certificationFail();

        } catch (Exception exception) {
            exception.printStackTrace();
            return ResponseDto.databaseError();

        }

        return CheckCertificationResponseDto.success();
    }

    @Override
    public ResponseEntity<? super SignUpResponseDto> signUp(SignUpRequestDto dto) {

        try {

            String userId = dto.getId();

            boolean isExistId = userRepository.existsByUserId(userId);
            if (isExistId)
                return SignUpResponseDto.duplicateId();

            String email = dto.getEmail();
            String certificationNumber = dto.getCertificationNumber();
            CertificationEntity certificationEntity = certificationRepository.findByUserId(userId);
            boolean isMatched = certificationEntity.getEmail().equals(email) &&
                    certificationEntity.getCertificationNumber().equals(certificationNumber);
            if (!isMatched)
                return SignUpResponseDto.certificationFail();

            String password = dto.getPassword();
            String encodedPassword = passwordEncoder.encode(password);
            dto.setPassword(encodedPassword);

            UserEntity userEntity = new UserEntity(dto);
            userRepository.save(userEntity);

            certificationRepository.deleteByUserId(userId);

        } catch (Exception exception) {
            exception.printStackTrace();
            return ResponseDto.databaseError();
        }

        return SignUpResponseDto.success();

    }

    public ResponseEntity<? super SignInResponseDto> signIn(SignInRequestDto dto) {
        String token = null;
        String userId = null;

        try {
            userId = dto.getId();
            UserEntity userEntity = userRepository.findByUserId(userId);
            if (userEntity == null)
                return SignInResponseDto.signInFail();

            String password = dto.getPassword();
            String encodedPassword = userEntity.getPassword();
            boolean isMatched = passwordEncoder.matches(password, encodedPassword);
            if (!isMatched)
                return SignInResponseDto.signInFail();

            token = jwtProvider.create(userId);

            userId = userEntity.getUserId();

        } catch (Exception exception) {
            exception.printStackTrace();
            return ResponseDto.databaseError();
        }
        return SignInResponseDto.success(token, userId);
    }
}
