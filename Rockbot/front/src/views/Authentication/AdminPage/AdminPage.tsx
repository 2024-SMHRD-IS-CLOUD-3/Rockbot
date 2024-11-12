import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useCookies } from 'react-cookie';
import { Unlock, Moon, Sun } from 'lucide-react';
import axios from 'axios';
import './AdminPage.css';

interface User {
    userId: string;
    name: string;
    departmentId: number;
}

interface TravelRequest {
    requestId: number;
    userId: string;
    name: string;
    departmentId: number;
    destination: string;
    travelDate: string;
    returnDate: string;
    reason: string;
    status: 'Pending' | 'Approved' | 'Rejected';
    submissionDate: string;
    fadingOut?: boolean;
}

// 부서 ID에 따른 부서 이름 설정 함수
const getDepartmentName = (departmentId: number) => {
    switch (departmentId) {
        case 1:
            return '임시부서';
        case 2:
            return '인사부서';
        case 3:
            return '편성부서';
        case 4:
            return '제작부서';
        default:
            return '알 수 없음';
    }
};

// 날짜 변환
const formatDate = (dateString: string) => {
    if (!dateString) return "날짜 정보 없음";
    try {
        const date = new Date(dateString);
        if (isNaN(date.getTime())) throw new Error("유효하지 않은 날짜 형식");

        const formatter = new Intl.DateTimeFormat('ko-KR', {
            year: 'numeric',
            month: 'long',
            day: 'numeric',
            hour: 'numeric',
            minute: 'numeric',
            second: 'numeric'
        });
        return formatter.format(date);
    } catch (error) {
        console.error("날짜 변환 오류:", error);
        return "유효하지 않은 날짜";
    }
};

const AdminPage: React.FC = () => {
    const [travelRequests, setTravelRequests] = useState<TravelRequest[]>([]);
    const [users, setUsers] = useState<User[]>([]);
    const [cookies, , removeCookie] = useCookies(['name', 'accessToken']);
    const navigate = useNavigate();
    const [selectedMenu, setSelectedMenu] = useState<'travelList' | 'travelPending' | 'travelRequests' | 'users'>('travelList');
    const [darkMode, setDarkMode] = useState<boolean>(false);

    const approvedRequests = travelRequests.filter(
        request => request.status === 'Approved' && new Date(request.returnDate) >= new Date()
    );

    const rejectedRequests = travelRequests.filter(request => request.status === 'Rejected');
    const pendingRequests = travelRequests.filter(request => request.status === 'Pending');

    // 복귀일까지 남은 일 수 계산 함수
    const calculateDaysUntilReturn = (returnDate: string) => {
        const today = new Date();
        const returnDay = new Date(returnDate);
        const timeDifference = returnDay.getTime() - today.getTime();
        const daysUntilReturn = Math.ceil(timeDifference / (1000 * 60 * 60 * 24));

        return daysUntilReturn > 0 ? `${daysUntilReturn}일 전` : '오늘 복귀 예정';
    };

    useEffect(() => {

        const fetchTravelRequests = async () => {
            try {
                const response = await axios.get('http://localhost:4040/api/v1/auth/travel-requests', {
                    headers: {
                        Authorization: `Bearer ${cookies.accessToken}`,
                    },
                });
                console.log('Fetched travel requests:', response.data);
                setTravelRequests(response.data);
            } catch (error) {
                console.error('출장 요청 목록을 가져오는 중 오류 발생:', error);
            }
        };

        const fetchUsers = async () => {
            try {
                const response = await axios.get('http://localhost:4040/api/v1/auth/all', {
                    headers: {
                        Authorization: `Bearer ${cookies.accessToken}`,
                    },
                });
                setUsers(response.data);
            } catch (error) {
                console.error('회원 정보를 가져오는 중 오류 발생:', error);
            }
        };

        fetchTravelRequests();
        fetchUsers();
    }, [cookies.accessToken]);

    const sendNotification = async (userId: string, message: string) => {
        console.log("Sending notification to userId:", userId, "with message:", message); // 확인용 로그
        try {
            await axios.post(
                'http://localhost:4040/api/v1/auth/notifications',
                { userId, message, status: 'Unread' },
                {
                    headers: {
                        Authorization: `Bearer ${cookies.accessToken}`,
                    },
                }
            );
        } catch (error) {
            console.error('알림 전송 중 오류 발생:', error);
        }
    };

// 출장 요청 승인 함수 수정
const handleApprove = async (id: number, userId: string, submissionDate: string, destination: string) => {
    try {
        await axios.put(
            `http://localhost:4040/api/v1/auth/travel-requests/${id}/status`,
            { status: 'Approved' },
            {
                headers: {
                    Authorization: `Bearer ${cookies.accessToken}`,
                },
            }
        );

        // 승인 후 알림 전송
        await sendNotification(userId, `${formatDate(submissionDate)} 신청한 ${destination} 출장 요청이 승인되었습니다.`);

        // 승인된 요청을 UI에서 승인 상태로 업데이트
        setTravelRequests((prevRequests) =>
            prevRequests.map((request) =>
                request.requestId === id ? { ...request, status: 'Approved' } : request
            )
        );

    } catch (error) {
        console.error('출장 요청 승인 중 오류 발생:', error);
    }
};

// 출장 요청 거절 함수 수정
const handleReject = async (id: number, userId: string, submissionDate: string, destination: string) => {
    try {
        await axios.put(
            `http://localhost:4040/api/v1/auth/travel-requests/${id}/status`,
            { status: 'Rejected' },
            {
                headers: {
                    Authorization: `Bearer ${cookies.accessToken}`,
                },
            }
        );

        // 거절 후 알림 전송
        await sendNotification(userId, `${formatDate(submissionDate)} 신청한 ${destination} 출장 요청이 거절되었습니다.`);

        // 거절된 요청을 UI에서 거절 상태로 업데이트
        setTravelRequests((prevRequests) =>
            prevRequests.map((request) =>
                request.requestId === id ? { ...request, status: 'Rejected' } : request
            )
        );

    } catch (error) {
        console.error('출장 요청 거절 중 오류 발생:', error);
    }
};

    const handleLogout = () => {
        removeCookie('name', { path: '/' });
        removeCookie('accessToken', { path: '/' });
        localStorage.removeItem('userInfo');
        navigate('/auth/sign-in');
    };

    const handleDepartmentChange = async (userId: string, newDepartmentId: number) => {
        try {
            const token = cookies.accessToken;
    
            console.log("User ID:", userId);
            console.log("New Department ID:", newDepartmentId);
    
            if (!token) {
                alert("로그인이 필요합니다.");
                return;
            }
    
            // 부서 이름을 가져오기
            const departmentName = getDepartmentName(newDepartmentId);
    
            console.log("Sending request with User ID:", userId, "and Department ID:", newDepartmentId);
    
            // 백엔드 서버로 PUT 요청을 보내 부서 ID를 변경
            await axios.put(
                `http://localhost:4040/api/v1/auth/${userId}/department`,
                { departmentId: newDepartmentId },
                {
                    headers: {
                        Authorization: `Bearer ${token}`,
                    },
                }
            );
    
            // 부서 변경 후 프론트엔드 상태 업데이트
            setUsers((prevUsers) =>
                prevUsers.map((user) =>
                    user.userId === userId ? { ...user, departmentId: newDepartmentId } : user
                )
            );
    
            // 부서 변경 후 알림 전송
            await sendNotification(userId, `부서가 ${departmentName}로 변경되었습니다.`);
    
            alert("부서가 성공적으로 변경되었습니다.");
        } catch (error) {
            console.error("부서 변경 중 오류 발생:", error);
            alert("부서 변경에 실패했습니다.");
        }
    };

    const toggleDarkMode = () => {
        setDarkMode(!darkMode);
    };

    return (
        <div className={`flex h-screen ${darkMode ? 'bg-dark-mode text-light' : 'bg-light-mode text-dark'}`}>
            {/* 사이드바 */}
            <div className="sidebar-container">
                <div className="sidebar-menu">
                    <button
                        className={`menu-item ${selectedMenu === 'travelList' ? 'active' : ''}`}
                        onClick={() => setSelectedMenu('travelList')}
                    >
                        출장 목록
                    </button>
                    <button
                        className={`menu-item ${selectedMenu === 'travelPending' ? 'active' : ''}`}
                        onClick={() => setSelectedMenu('travelPending')}
                    >
                        결재 대기
                    </button>
                    <button
                        className={`menu-item ${selectedMenu === 'travelRequests' ? 'active' : ''}`}
                        onClick={() => setSelectedMenu('travelRequests')}
                    >
                        결재 목록
                    </button>
                    <button
                        className={`menu-item ${selectedMenu === 'users' ? 'active' : ''}`}
                        onClick={() => setSelectedMenu('users')}
                    >
                        직원 목록
                    </button>
                </div>

                <div className="sidebar-bottom-buttons">
                    <button className="button dark-mode-toggle-button" onClick={toggleDarkMode}>
                        {darkMode ? <Sun size={16} className="icon-spacing" /> : <Moon size={15} className="icon-spacing" />}
                        {darkMode ? '라이트 모드' : '다크 모드'}
                    </button>
                    <button onClick={handleLogout} className="button navigate-button">
                        <Unlock size={15} className="icon-spacing" /> 로그아웃
                    </button>
                </div>
            </div>

            {selectedMenu === 'travelPending' && (
                <div className="pending-list-container">
                    {pendingRequests.map((request) => (
                        <div key={request.requestId} className="trip-request">
                            <div className="trip-info">
                                <p className="trip-name">이름 : {request.name}</p>
                                <p className="trip-department">부서 : {getDepartmentName(request.departmentId)}</p>
                                <p className="trip-destination">목적지 : {request.destination}</p>
                                <p className="trip-dates">
                                    출장 날짜 : {formatDate(request.travelDate)} ~ {formatDate(request.returnDate)}까지
                                </p>
                                <p className="trip-reason">사유 : {request.reason}</p>
                                <p className="trip-submission-date">신청 날짜 : {formatDate(request.submissionDate)}</p>
                            </div>
                            <div className="trip-actions">
                            <button 
    className="approve-button" 
    onClick={() => handleApprove(request.requestId, request.userId, request.submissionDate, request.destination)}>
    ✔ 승인
</button>
<button 
    className="reject-button" 
    onClick={() => handleReject(request.requestId, request.userId, request.submissionDate, request.destination)}>
    ✖ 거절
</button>

                            </div>

                        </div>
                    ))}
                </div>
            )}

            {/* 출장 목록 */}
            {selectedMenu === 'travelList' && (
                <div className="travel-list-container">
                    {approvedRequests.map((request) => (
                        <div key={request.requestId} className="trip-item">
                            <div className="trip-info">
                                <p className="trip-name">이름 : {request.name}</p>
                                <p className="trip-department">부서 ID : {getDepartmentName(request.departmentId)}</p>
                                <p className="trip-destination">목적지 : {request.destination}</p>
                                <p className="trip-dates">
                                    출장 날짜 : {formatDate(request.travelDate)} ~ {formatDate(request.returnDate)}까지
                                </p>
                                <p className="trip-reason">사유 : {request.reason}</p>
                                <p className="trip-submission-date">신청 날짜 : {formatDate(request.submissionDate)}</p>
                                <p className="trip-return-days"> ＊복귀 {calculateDaysUntilReturn(request.returnDate)}</p>
                            </div>
                        </div>
                    ))}
                </div>
            )}

            {selectedMenu === 'travelRequests' && (
                <div className="request-list-container">
                    {approvedRequests.map((request) => (
                        <div key={request.requestId} className="trip-request approved">
                            <div className="trip-info">
                                <p className="trip-name">이름 : {request.name}</p>
                                <p className="trip-department">부서 : {getDepartmentName(request.departmentId)}</p>
                                <p className="trip-destination">목적지 : {request.destination}</p>
                                <p className="trip-dates">
                                    출장 날짜 : {formatDate(request.travelDate)} ~ {formatDate(request.returnDate)}까지
                                </p>
                                <p className="trip-reason">사유 : {request.reason}</p>
                                <p className="trip-submission-date">신청 날짜 : {formatDate(request.submissionDate)}</p>
                                <p className="trip-status">＊승인됨</p>
                            </div>
                        </div>
                    ))}
                    {rejectedRequests.map((request) => (
                        <div key={request.requestId} className="trip-request rejected">
                            <div className="trip-info">
                                <p className="trip-name">이름 : {request.name}</p>
                                <p className="trip-department">부서 : {getDepartmentName(request.departmentId)}</p>
                                <p className="trip-destination">목적지 : {request.destination}</p>
                                <p className="trip-dates">
                                    출장 날짜 : {formatDate(request.travelDate)} ~ {formatDate(request.returnDate)}까지
                                </p>
                                <p className="trip-reason">사유 : {request.reason}</p>
                                <p className="trip-submission-date">신청 날짜 : {formatDate(request.submissionDate)}</p>
                                <p className="trip-status">＊거절됨</p>
                            </div>
                        </div>
                    ))}
                </div>
            )}

            {selectedMenu === 'users' && (
                <div className="user-list-container">
                    {users
                        .filter((user) => user.userId !== 'Admin') // 관리자 계정 제외
                        .map((user) => (
                            <div key={user.userId} className="user-info">
                                <p>이름 : {user.name}</p>
                                <p>부서 : {getDepartmentName(user.departmentId)}</p>
                                <label>
                                    부서 변경</label>
                                <select
                                    className="custom-select"
                                    value={user.departmentId}
                                    onChange={(e) => handleDepartmentChange(user.userId, parseInt(e.target.value))}
                                >
                                    <option value={1}>임시부서</option>
                                    <option value={2}>인사부서</option>
                                    <option value={3}>편성부서</option>
                                    <option value={4}>제작부서</option>
                                </select>
                            </div>
                        ))}
                </div>
            )}
        </div>
    );
};

export default AdminPage;