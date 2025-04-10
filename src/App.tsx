import React, { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import ChatInterface from "./components/ChatInterface";
import Dashboard from "./components/Dashboard";
import { FaChartBar, FaComments } from "react-icons/fa";

function App() {
  const [showDashboard, setShowDashboard] = useState(false);
  const [responses, setResponses] = useState({});

  const handleChatComplete = (userResponses: React.SetStateAction<{}>) => {
    setResponses(userResponses);
    setShowDashboard(true);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-purple-50">
      <nav className="bg-white shadow-lg p-4">
        <div className="container mx-auto flex justify-between items-center">
          <h1 className="text-2xl font-bold text-gray-800">
            Mental Health Assessment
          </h1>
          <div className="flex gap-4">
            <button
              onClick={() => setShowDashboard(false)}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg ${
                !showDashboard
                  ? "bg-blue-500 text-white"
                  : "bg-gray-100 text-gray-600"
              }`}
            >
              <FaComments /> Chat
            </button>
            <button
              onClick={() => setShowDashboard(true)}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg ${
                showDashboard
                  ? "bg-blue-500 text-white"
                  : "bg-gray-100 text-gray-600"
              }`}
              disabled={Object.keys(responses).length === 0}
            >
              <FaChartBar /> Dashboard
            </button>
          </div>
        </div>
      </nav>

      <div className="container mx-auto p-4">
        <AnimatePresence mode="wait">
          {!showDashboard ? (
            <motion.div
              key="chat"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.3 }}
            >
              <ChatInterface onComplete={handleChatComplete} />
            </motion.div>
          ) : (
            <motion.div
              key="dashboard"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.3 }}
            >
              <Dashboard responses={responses} />
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}

export default App;
