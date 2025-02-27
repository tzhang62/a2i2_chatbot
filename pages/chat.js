import { useState, useEffect } from 'react';
import { useRouter } from 'next/router';
import axios from 'axios';

export default function Chat() {
  const router = useRouter();
  const { person } = router.query;
  const [messages, setMessages] = useState([]);
  const [userInput, setUserInput] = useState("");
  const [isAuto, setIsAuto] = useState(false);

  // When a town person is selected and auto mode is active, trigger auto chat
  useEffect(() => {
    if (isAuto && person) {
      autoGenerateChat();
    }
  }, [isAuto, person]);

  const sendMessage = async () => {
    if (!userInput.trim()) return;

    // Add the user's message to the chat
    setMessages(prev => [...prev, { sender: "You", text: userInput }]);

    try {
      const response = await axios.post("http://localhost:8000/chat", {
        townPerson: person,
        userInput: userInput,
        mode: "interactive"
      });
      // Add the operator's response to the chat
      setMessages(prev => [...prev, { sender: "Operator", text: response.data.response }]);
    } catch (error) {
      console.error("Error sending message:", error);
    }

    setUserInput("");
  };

  const autoGenerateChat = async () => {
    try {
      const response = await axios.post("http://localhost:8000/chat", {
        townPerson: person,
        userInput: "",
        mode: "auto"
      });

      // Assume the backend returns a newline-separated string of conversation lines
      const lines = response.data.response.split("\n");
      const chatHistory = lines.map((line, index) => ({
        sender: index % 2 === 0 ? "Operator" : "TownPerson",
        text: line.trim()
      }));

      setMessages(chatHistory);
    } catch (error) {
      console.error("Error generating auto chat:", error);
    }
  };

  return (
    <div className="p-6">
      <h1 className="text-xl font-bold">
        Chat with Fire Department Operator as {person}
      </h1>
      <div className="flex space-x-4 mt-4">
        <button
          onClick={() => setIsAuto(true)}
          className="bg-gray-600 text-white px-4 py-2 rounded"
        >
          Auto Chat
        </button>
        <button
          onClick={() => setIsAuto(false)}
          className="bg-gray-600 text-white px-4 py-2 rounded"
        >
          Interactive Mode
        </button>
      </div>
      <div className="mt-4 bg-gray-100 p-4 rounded-lg h-64 overflow-y-auto">
        {messages.map((msg, index) => (
          <p key={index} className={`p-2 ${msg.sender === "You" ? "text-right" : "text-left"}`}>
            <strong>{msg.sender}:</strong> {msg.text}
          </p>
        ))}
      </div>
      {!isAuto && (
        <div className="mt-4 flex">
          <input
            type="text"
            value={userInput}
            onChange={(e) => setUserInput(e.target.value)}
            placeholder="Type your message..."
            className="flex-1 p-2 border border-gray-400 rounded"
          />
          <button
            onClick={sendMessage}
            className="ml-2 bg-blue-500 text-white px-4 py-2 rounded"
          >
            Send
          </button>
        </div>
      )}
    </div>
  );
}
