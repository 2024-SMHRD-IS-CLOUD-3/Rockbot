package com.rockbot.back.controller;

import java.util.List;
import java.util.Map;
import java.util.Optional;

import javax.management.Notification;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.PutMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import com.rockbot.back.dto.request.auth.CheckCertificationRequestDto;
import com.rockbot.back.dto.request.auth.EmailCertificationRequestDto;
import com.rockbot.back.dto.request.auth.IdCheckRequestDto;
import com.rockbot.back.dto.request.auth.NotificationRequestDto;
import com.rockbot.back.dto.request.auth.SignInRequestDto;
import com.rockbot.back.dto.request.auth.SignUpRequestDto;
import com.rockbot.back.dto.response.auth.CheckCertificationResponseDto;
import com.rockbot.back.dto.response.auth.EmailCertificationResponseDto;
import com.rockbot.back.dto.response.auth.IdCheckResponseDto;
import com.rockbot.back.dto.response.auth.SignInResponseDto;
import com.rockbot.back.dto.response.auth.SignUpResponseDto;
import com.rockbot.back.entity.NotificationEntity; // 올바른 Notification 엔터티 임포트
import com.rockbot.back.entity.TravelEntity;
import com.rockbot.back.entity.UserEntity;
import com.rockbot.back.repository.NotificationRepository;
import com.rockbot.back.repository.TravelRequestRepository;
import com.rockbot.back.repository.UserRepository;
import com.rockbot.back.service.AuthService;

import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;

@RestController
@RequestMapping("/api/v1/auth")
@RequiredArgsConstructor
public class AuthController {

    private final AuthService authService;

    @Autowired
    private UserRepository userRepository;

    @Autowired
    private TravelRequestRepository travelRequestRepository;

    @Autowired
    private NotificationRepository notificationRepository;

    @PostMapping("/id-check")
    public ResponseEntity<? super IdCheckResponseDto> idCheck(
            @RequestBody @Valid IdCheckRequestDto requestBody) {
        ResponseEntity<? super IdCheckResponseDto> response = authService.idCheck(requestBody);
        return response;
    }

    @PostMapping("/email-certification")
    public ResponseEntity<? super EmailCertificationResponseDto> emailCertification(
            @RequestBody @Valid EmailCertificationRequestDto requestBody) {
        ResponseEntity<? super EmailCertificationResponseDto> response = authService.emailCertification(requestBody);
        return response;
    }

    @PostMapping("/check-certification")
    public ResponseEntity<? super CheckCertificationResponseDto> checkCertification(
            @RequestBody @Valid CheckCertificationRequestDto requestBody) {
        ResponseEntity<? super CheckCertificationResponseDto> response = authService.checkCertification(requestBody);
        return response;
    }

    @PostMapping("/sign-up")
    public ResponseEntity<? super SignUpResponseDto> signUp(
            @RequestBody @Valid SignUpRequestDto requestBody) {
        ResponseEntity<? super SignUpResponseDto> response = authService.signUp(requestBody);
        return response;
    }

    @PostMapping("/sign-in")
    public ResponseEntity<? super SignInResponseDto> signIn(
            @RequestBody @Valid SignInRequestDto requestBody) {
        ResponseEntity<? super SignInResponseDto> response = authService.signIn(requestBody);
        return response;
    }

    // 가입회원 정보 띄우기
    @GetMapping("/all")
    public List<UserEntity> getAllUsers() {
        return userRepository.findAll();
    }

    // 부서 변경 엔드포인트
    @PutMapping("/{userId}/department")
    public ResponseEntity<?> updateUserDepartment(@PathVariable String userId,
            @RequestBody Map<String, Integer> request) {
        try {
            Integer newDepartmentId = request.get("departmentId");
            UserEntity user = userRepository.findById(userId)
                    .orElseThrow(() -> new RuntimeException("사용자를 찾을 수 없습니다."));

            // 부서 ID 업데이트
            user.setDepartmentId(newDepartmentId);
            userRepository.save(user);

            return ResponseEntity.ok("부서가 성공적으로 변경되었습니다.");
        } catch (Exception e) {
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body("부서 변경에 실패했습니다.");
        }
    }

    // 모든 출장 요청 목록 가져오기
    @GetMapping("/travel-requests")
    public ResponseEntity<List<TravelEntity>> getAllTravelRequests() {
        List<TravelEntity> travelRequests = travelRequestRepository.findAll();
        return ResponseEntity.ok(travelRequests);
    }

    // 특정 출장 요청의 상태 업데이트
    @PutMapping("/travel-requests/{requestId}/status")
    public ResponseEntity<String> updateTravelRequestStatus(
            @PathVariable Long requestId,
            @RequestBody Map<String, String> request) {
        Optional<TravelEntity> travelRequestOptional = travelRequestRepository.findById(requestId);

        if (!travelRequestOptional.isPresent()) {
            return ResponseEntity.notFound().build();
        }

        String newStatus = request.get("status");
        TravelEntity travelRequest = travelRequestOptional.get();
        travelRequest.setStatus(newStatus);
        travelRequestRepository.save(travelRequest);

        return ResponseEntity.ok("출장 요청 상태가 업데이트되었습니다.");
    }

    @PostMapping("/notifications")
    public ResponseEntity<?> createNotification(@RequestBody NotificationRequestDto notificationRequest) {
        try {
            NotificationEntity notification = new NotificationEntity();
            notification.setUserId(notificationRequest.getUserId());
            notification.setMessage(notificationRequest.getMessage());
            notification.setStatus("Unread");
            notificationRepository.save(notification);

            return ResponseEntity.ok("알림이 저장되었습니다.");
        } catch (Exception e) {
            e.printStackTrace(); // 오류 로그 출력
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body("알림 저장 중 오류 발생");
        }
    }

    @GetMapping("/notifications/{userId}")
    public ResponseEntity<List<NotificationEntity>> getUnreadNotifications(@PathVariable String userId) {
        try {
            List<NotificationEntity> notifications = notificationRepository.findByUserIdAndStatus(userId, "Unread");
            return ResponseEntity.ok(notifications);
        } catch (Exception e) {
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body(null);
        }
    }

    @PutMapping("/notifications/{notificationId}")
    public ResponseEntity<?> markNotificationAsRead(@PathVariable Integer notificationId) {
        try {
            Optional<NotificationEntity> notificationOptional = notificationRepository.findById(notificationId);
            if (notificationOptional.isPresent()) {
                NotificationEntity notification = notificationOptional.get();
                notification.setStatus("Read");
                notificationRepository.save(notification);
                return ResponseEntity.ok("알림이 읽음으로 표시되었습니다.");
            } else {
                return ResponseEntity.status(HttpStatus.NOT_FOUND).body("해당 알림을 찾을 수 없습니다.");
            }
        } catch (Exception e) {
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body("알림 상태 업데이트 중 오류 발생");
        }
    }
}
