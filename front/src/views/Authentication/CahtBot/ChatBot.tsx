import React, { useState, useEffect, useRef } from 'react';
import { useParams,useLocation, useNavigate } from 'react-router-dom';
import axios from 'axios';
import './ChatBot.css';
import { Unlock, Moon, Sun, Send, UserRound } from 'lucide-react';

interface ProfileInfo {
  password?: string;
  userId?: string;
}

const ChatBot: React.FC = () => {
  const [question, setQuestion] = useState<string>('');
  const [chatHistory, setChatHistory] = useState<Array<{ sender: string; message: string }>>([
    { sender: 'bot', message: '무엇을 도와드릴까요?' },
  ]);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [darkMode, setDarkMode] = useState<boolean>(false);
  const [currentMode, setCurrentMode] = useState<string>('업무');
  const [showProfileSettings, setShowProfileSettings] = useState<boolean>(false);
  const [profileInfo, setProfileInfo] = useState<ProfileInfo>({
    userId: ''
  });
  const { userId } = useParams<{ userId: string }>();
  const loc = useLocation();
  const chatContainerRef = useRef<HTMLDivElement | null>(null);
  const inputRef = useRef<HTMLInputElement | null>(null);


  const navigate = useNavigate(); // useNavigate 훅 추가

  // 로그인 정보 출력
  useEffect(() => {
    
    const searchParams = new URLSearchParams(loc.search);
    const userIdFromURL = searchParams.get('userId');
    
    const storedUserInfo = localStorage.getItem('userInfo');
    if (userIdFromURL) {
      // 쿼리스트링에 userId가 있을 경우, 해당 값으로 설정
      setProfileInfo({ userId: userIdFromURL });
    } else if (storedUserInfo) {
      setProfileInfo(JSON.parse(storedUserInfo));
    } else  {
      setProfileInfo({ userId: 'Guest' }); // 기본값 설정
    }
  }, [loc]);

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setQuestion(e.target.value);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      handleSend(); // 엔터 키가 눌리면 handleSend 함수 실행
    }
  };

  const handleSend = async () => {
    if (!question.trim()) return;

    setIsLoading(true);
    const newChatHistory = [...chatHistory, { sender: 'user', message: question }];
    setChatHistory(newChatHistory);

    try {
      // Flask 서버에 질문을 보냅니다.
      const response = await axios.post('http://localhost:5000/ask', {
        question: question,
      });

      // 서버의 응답을 받아 chatHistory에 추가합니다.
      setChatHistory((prevHistory) => [
        ...prevHistory,
        { sender: 'bot', message: response.data.answer }, // 서버에서 받아온 응답 사용
      ]);
    } catch (error) {
      // 서버와의 연결이 실패했을 경우
      setChatHistory((prevHistory) => [
        ...prevHistory,
        { sender: 'bot', message: '응답을 가져오는 데 실패했습니다.' },
      ]);
    } finally {
      setIsLoading(false);
    }
    setQuestion('');

    // 메시지 전송 후 입력 필드에 포커스를 유지
    if (inputRef.current && !isLoading) {
      inputRef.current.focus();
    }
  };

  useEffect(() => {
    if (inputRef.current && !isLoading) {
      inputRef.current.focus(); // question 상태가 변경되면 입력 필드에 포커스를 설정합니다.
    }
  }, [question]);

  // 새로운 메시지 추가될 때마다 스크롤을 맨 아래로 내림
  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  }, [chatHistory]);

  const toggleDarkMode = () => setDarkMode(!darkMode);

  const toggleProfileSettings = () => {
    setShowProfileSettings(!showProfileSettings);
  };

  return (
    <div className={`flex h-screen ${darkMode ? 'bg-dark-mode text-light' : 'bg-light-mode text-dark'}`}>
       {/* 사이드바 */}
       <div className="sidebar-container">
        {/* 상단에 위치할 요소들 */}
        <div>
          <div className="profile-container mb-4">
            {profileInfo.userId && (
              <p className="profile-user-id">{profileInfo.userId}</p> // 사용자 ID 표시
            )}
          </div>
          {/* <button onClick={toggleProfileSettings} className="button profile-settings-button">
            <UserRound size={15} className="icon-spacing" /> 비밀번호 변경
          </button> */}
        </div>

        {/* 하단에 위치시킬 버튼들 */}
        <div className="sidebar-bottom-buttons">
          <button onClick={toggleDarkMode} className="button dark-mode-toggle-button">
            {darkMode ? <Sun size={16} className="icon-spacing" /> : <Moon size={15} className="icon-spacing" />}
            {darkMode ? '라이트 모드' : '다크 모드'}
          </button>
          <button onClick={() => navigate('/auth/sign-in')} className="button navigate-button">
            <Unlock size={15} className="icon-spacing" />로그아웃
          </button>
        </div>
      </div>

      {/* 대화창 */}
      <div className="chat-window-container">
        {showProfileSettings && (
          <div className="profile-settings-container">
            {/* <h3 className="profile-settings-header">비밀번호 변경</h3>
            <input
              type="password"
              value={profileInfo.password || ''}
              onChange={(e) => setProfileInfo({ ...profileInfo, password: e.target.value })}
              className="input-field-password"
              placeholder="변경하실 비밀번호를 입력하세요"
            />
            <button onClick={toggleProfileSettings} className="button profile-settings-save-button">
              저장
            </button> */}
          </div>
        )}

<div className="chat-messages-container" ref={chatContainerRef}>
          {chatHistory.map((chat, index) => (
            <div
              key={index}
              className={`message-container ${chat.sender === 'user' ? 'user-message' : 'bot-message'}`}
            >
              <div className={`message-bubble ${chat.sender === 'user' ? 'user-bubble' : 'bot-bubble'}`}>{
                chat.message
              }</div>
            </div>
          ))}
        </div>
        <div className="chat-input-container">
          <div className="input-group">
            <input
              type="text"
              value={question}
              onChange={handleInputChange}
              onKeyDown={handleKeyDown}
              ref={inputRef}
              className="input-field chat-input"
              placeholder="질문을 입력하세요..."
              disabled={isLoading}
            />
            <button onClick={handleSend} className="button send-button" disabled={isLoading}>
              <Send size={20} />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ChatBot;
