package com.rockbot.back.repository;

import com.rockbot.back.entity.NotificationEntity;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;
import java.util.List;

@Repository
public interface NotificationRepository extends JpaRepository<NotificationEntity, Integer> {
    List<NotificationEntity> findByUserId(String userId);

    // 사용자 ID와 상태를 기준으로 알림 조회
    List<NotificationEntity> findByUserIdAndStatus(String userId, String status);
}