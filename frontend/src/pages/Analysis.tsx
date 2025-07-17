import { useState, useRef, useEffect } from "react";
import { useLocation, useNavigate, useParams } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Textarea } from "@/components/ui/textarea";
import { Progress } from "@/components/ui/progress";
import { Separator } from "@/components/ui/separator";
import { 
  Upload, Mic, Video, Pause, Send, Bot, Calendar, TrendingUp, 
  Heart, AlertTriangle, Smile, Frown, MessageSquare, Eye 
} from "lucide-react";
import Navigation from "@/components/shared/Navigation";
import Footer from "@/components/shared/Footer";
import { useAudioRecording } from "@/hooks/useAudioRecording";
import { extractAudioFromVideo} from "@/utils/audioUtils";
import { useToast } from "@/hooks/use-toast";
import ReactMarkdown from 'react-markdown';

interface Pet {
  id: string;
  name: string;
  breed: string;
  age: string;
  photo: string;
  description?: string;
}

interface EmotionHistory {
  id: string;
  date: string;
  emotion: string;
  confidence: number;
  audioType: string;
  recommendation: string;
}

const Analysis = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const { petId } = useParams();
  const pet = location.state?.pet as Pet;
  const { toast } = useToast();
  
  const [emotionHistory, setEmotionHistory] = useState<EmotionHistory[]>([
  ]);
  
  const [showAllHistory, setShowAllHistory] = useState(false);

  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResult, setAnalysisResult] = useState<EmotionHistory | null>(null);
  const [chatMessage, setChatMessage] = useState("");
  const [chatHistory, setChatHistory] = useState<Array<{role: 'user' | 'assistant', message: string}>>([
    {
        role: 'assistant',
        message: `Hello! I'm your AI assistant. How can I help you with ${pet.name}?`
    }
  ]);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const videoInputRef = useRef<HTMLInputElement>(null);
  
  const { 
    isRecording, 
    recordedBlob, 
    startRecording, 
    stopRecording, 
    error: recordingError 
  } = useAudioRecording();

  useEffect(() => {
    if (recordingError) {
      toast({
        title: "Recording Error",
        description: recordingError,
        variant: "destructive"
      });
    }
  }, [recordingError, toast]);

  useEffect(() => {
    if (recordedBlob) {
      handleAudioAnalysis(recordedBlob, 'Live Recording');
    }
  }, [recordedBlob]);

  if (!pet) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <div className="text-center">
          <h2 className="text-2xl font-bold mb-4">Pet not found</h2>
          <Button onClick={() => navigate('/home')}>Return to Home</Button>
        </div>
      </div>
    );
  }

  const handleAudioAnalysis = async (audioBlob: Blob, audioType: string) => {
    setIsAnalyzing(true);
    
    try {
      const formData = new FormData();
        formData.append("audioOrVideoFile", audioBlob, "audio.wav");  // fake filename is fine
        formData.append("fileType", audioType === ("video") ? "video" : "audio");
        formData.append("Pet.Name", pet.name);
        formData.append("Pet.Breed", pet.breed);
        formData.append("Pet.Age", pet.age);
        formData.append("Pet.description?", pet.description || "");

        const res = await fetch("/api/analyze-cat-emotion", {
        method: "POST",
        body: formData,
        });

        if (!res.ok) throw new Error("API call failed");

        const data = await res.json();
      
        const result: EmotionHistory = {
            id: Date.now().toString(),
            date: new Date().toLocaleString(),
            emotion: data.analysisResult.emotions.primary,
            confidence: Math.round(data.analysisResult.emotions.confidence * 100),
            audioType,
            recommendation: data.initialChatbotMessage
        };
      
      setAnalysisResult(result);
      
      // Add to emotion history (keep only last 5 + new one)
      setEmotionHistory(prev => {
        const updated = [result, ...prev];
        return updated.slice(0, 5); // Keep only most recent 5
      });
      
      toast({
        title: "Analysis Complete",
        description: `${pet?.name} seems to be feeling ${result.emotion.toLowerCase()}!`
      });
    } catch (error) {
      console.error('Analysis error:', error);
      toast({
        title: "Analysis Failed",
        description: "Failed to analyze the audio. Please try again.",
        variant: "destructive"
      });
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleFileUpload = async (file: File, type: 'audio' | 'video') => {
    setSelectedFile(file);
    
    try {
      let audioBlob: Blob;
      audioBlob = file;
      await handleAudioAnalysis(audioBlob, type);
    } catch (error) {
      console.error('File processing error:', error);
      toast({
        title: "Processing Failed",
        description: "Failed to process the file. Please try again.",
        variant: "destructive"
      });
    }
  };

  const handleRecordingToggle = () => {
    if (isRecording) {
      stopRecording();
    } else {
      startRecording();
    }
  };

  const handleChatSubmit = async () => {
    if (!chatMessage.trim()) return;

    // Add user's message immediately
    setChatHistory(prev => [...prev, { role: 'user', message: chatMessage }]);

    try {
        const res = await fetch("/api/chat", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({
            message: chatMessage,
            petName: pet.name,
            petBreed: pet.breed,
            emotion: analysisResult?.emotion || "unknown",
            confidence: Math.round(analysisResult?.confidence),
        })
        });

        if (!res.ok) throw new Error("Chat API failed");

        const data = await res.json();
        const assistantMessage = data.reply || "Sorry, I couldn’t come up with a good answer.";
        setChatHistory(prev => [...prev, { role: 'assistant', message: assistantMessage }]);
    } catch (error) {
        console.error("Chat error:", error);
        setChatHistory(prev => [...prev, {
        role: 'assistant',
        message: "Oops! I couldn't get a response. Try again later."
        }]);
    } finally {
        setChatMessage("");
    }
    };


  const getEmotionColor = (emotion: string) => {
    const colors: Record<string, string> = {
      'Happy': 'bg-success',
      'Resting': 'bg-accent',
      'Angry': 'bg-destructive',
      'Fighting': 'bg-destructive',
      'Defense': 'bg-yellow-{50-950}',
      'Hunting': 'bg-accent',
      'MotherCall': 'bg-primary',
      'Mating': 'bg-primary',
      'Paining': 'bg-yellow-{50-950}',
      'Warning': 'bg-destructive',
    };
    return colors[emotion] || 'bg-muted';
  };

  const getEmotionIcon = (emotion: string) => {
    const icons: Record<string, JSX.Element> = {
      'Happy': <Smile className="h-4 w-4" />,
      'Resting': <Smile className="h-4 w-4" />,
      'Angry': <AlertTriangle className="h-4 w-4" />,
      'Fighting': <AlertTriangle className="h-4 w-4" />,
      'Defense': <AlertTriangle className="h-4 w-4" />,
      'Hunting': <TrendingUp className="h-4 w-4" />,
      'MotherCall': <MessageSquare className="h-4 w-4" />,
      'Mating': <Heart className="h-4 w-4" />,
      'Paining': <Frown className="h-4 w-4" />,
      'Warning': <AlertTriangle className="h-4 w-4" />,
    };
    return icons[emotion] || <MessageSquare className="h-4 w-4" />;
  };

  return (
    <div className="min-h-screen bg-background">
      <Navigation />

      {/* Main Content */}
      <main className="container mx-auto max-w-6xl px-4 py-8">
        <div className="grid lg:grid-cols-3 gap-8">
          {/* Left Column - Pet Profile & History */}
          <div className="lg:col-span-1 space-y-6">
            {/* Pet Profile */}
            <Card className="bg-gradient-to-br from-card to-accent/10 border-2 border-accent/20">
              <CardHeader>
                <CardTitle className="flex items-center gap-3">
                  <span className="text-3xl">{pet.photo}</span>
                  <div>
                    <h3 className="text-xl">{pet.name}</h3>
                    <p className="text-sm text-muted-foreground">{pet.breed}</p>
                  </div>
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium">Age:</span>
                  <Badge variant="secondary">{pet.age}</Badge>
                </div>
                {pet.description && (
                  <div>
                    <span className="text-sm font-medium">About:</span>
                    <p className="text-sm text-muted-foreground mt-1">{pet.description}</p>
                  </div>
                )}
              </CardContent>
            </Card>

            {/* Emotion History */}
            <Card className="bg-gradient-to-br from-card to-success/10 border-2 border-success/20">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Calendar className="h-5 w-5" />
                  Emotion History
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                {(showAllHistory ? emotionHistory : emotionHistory.slice(0, 5)).map((entry) => (
                  <div key={entry.id} className="space-y-2">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <div className={`p-1 rounded-full text-white ${getEmotionColor(entry.emotion)}`}>
                          {getEmotionIcon(entry.emotion)}
                        </div>
                        <span className="font-medium text-sm">{entry.emotion}</span>
                      </div>
                      <span className="text-xs text-muted-foreground">{entry.confidence}%</span>
                    </div>
                    <Progress value={entry.confidence} className="h-2" />
                    <p className="text-xs text-muted-foreground">{entry.date}</p>
                    <Separator />
                  </div>
                ))}
                
                {emotionHistory.length > 5 && (
                  <Button 
                    variant="outline" 
                    size="sm" 
                    className="w-full"
                    onClick={() => setShowAllHistory(!showAllHistory)}
                  >
                    <Eye className="mr-2 h-3 w-3" />
                    {showAllHistory ? 'Show Recent Only' : `View All (${emotionHistory.length})`}
                  </Button>
                )}
              </CardContent>
            </Card>
          </div>

          {/* Right Column - Analysis & Chat */}
          <div className="lg:col-span-2 space-y-6">
            {/* Upload Section */}
            <Card className="bg-gradient-to-br from-card to-lavender/10 border-2 border-lavender/20">
              <CardHeader>
                <CardTitle>Analyze {pet.name}'s Emotions</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid md:grid-cols-3 gap-4">
                  <Button
                    variant="outline"
                    className="h-24 flex flex-col gap-2 bg-accent/20 hover:bg-accent/30 border-accent"
                    onClick={() => fileInputRef.current?.click()}
                  >
                    <Upload className="h-6 w-6" />
                    Upload Audio
                  </Button>
                  
                  <Button
                    variant="outline"
                    className="h-24 flex flex-col gap-2 bg-success/20 hover:bg-success/30 border-success"
                    onClick={() => videoInputRef.current?.click()}
                  >
                    <Video className="h-6 w-6" />
                    Upload Video
                  </Button>
                  
                  <Button
                    variant={isRecording ? "destructive" : "outline"}
                    className={`h-24 flex flex-col gap-2 ${!isRecording ? 'bg-lavender/20 hover:bg-lavender/30 border-lavender' : ''}`}
                    onClick={handleRecordingToggle}
                    disabled={isAnalyzing}
                  >
                    {isRecording ? <Pause className="h-6 w-6" /> : <Mic className="h-6 w-6" />}
                    {isRecording ? "Stop Recording" : "Live Recording"}
                  </Button>
                </div>

                <input
                  ref={fileInputRef}
                  type="file"
                  accept="audio/*"
                  className="hidden"
                  onChange={(e) => e.target.files?.[0] && handleFileUpload(e.target.files[0], 'audio')}
                />
                
                <input
                  ref={videoInputRef}
                  type="file"
                  accept="video/*"
                  className="hidden"
                  onChange={(e) => e.target.files?.[0] && handleFileUpload(e.target.files[0], 'video')}
                />

                {selectedFile && (
                  <div className="p-4 bg-muted rounded-lg">
                    <p className="text-sm font-medium">Selected file: {selectedFile.name}</p>
                    <p className="text-xs text-muted-foreground">Ready for analysis</p>
                  </div>
                )}

                {isRecording && (
                  <div className="p-4 bg-destructive/10 rounded-lg border border-destructive/20">
                    <div className="flex items-center gap-2 mb-2">
                      <div className="animate-pulse rounded-full h-3 w-3 bg-destructive"></div>
                      <span className="text-sm font-medium text-destructive-foreground">Recording {pet.name}'s audio...</span>
                    </div>
                    <p className="text-xs text-muted-foreground">Speak clearly or let your cat vocalize near the microphone</p>
                  </div>
                )}

                {isAnalyzing && (
                  <div className="p-4 bg-accent/20 rounded-lg">
                    <div className="flex items-center gap-2 mb-2">
                      <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-primary"></div>
                      <span className="text-sm font-medium">Analyzing {pet.name}'s emotions...</span>
                    </div>
                    <Progress value={33} className="h-2" />
                  </div>
                )}
              </CardContent>
            </Card>

            {/* Analysis Results */}
            {analysisResult && (
              <Card className="bg-gradient-to-br from-card to-primary/10 border-2 border-primary/20">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    Analysis Results
                    <Badge className={getEmotionColor(analysisResult.emotion)}>
                      {analysisResult.emotion}
                    </Badge>
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="grid md:grid-cols-2 gap-4">
                    <div>
                      <p className="text-sm font-medium mb-1">Confidence Level</p>
                      <Progress value={analysisResult.confidence} className="h-3" />
                      <p className="text-xs text-muted-foreground mt-1">{analysisResult.confidence}%</p>
                    </div>
                    <div>
                      <p className="text-sm font-medium mb-1">Audio Type</p>
                      <p className="text-sm text-muted-foreground">{analysisResult.audioType}</p>
                    </div>
                  </div>
                  
                  <div className="p-4 bg-success/10 rounded-lg border border-success/20">
                    <h4 className="font-medium text-success-foreground mb-2">Recommendation</h4>
                    <p className="text-sm text-success-foreground">{analysisResult.recommendation}</p>
                  </div>
                </CardContent>
              </Card>
            )}

            {/* Chat Section */}
            <Card className="bg-gradient-to-br from-card to-secondary/20 border-2 border-secondary/30">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Bot className="h-5 w-5" />
                  Chat with AI Assistant
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4 mb-4 max-h-64 overflow-y-auto">
                  {chatHistory.map((message, index) => (
                    <div key={index} className={`flex gap-3 ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                      <div className={`p-3 rounded-lg max-w-xs ${
                        message.role === 'user' 
                          ? 'bg-primary text-primary-foreground ml-12' 
                          : 'bg-muted text-muted-foreground mr-12'
                      }`}>
                        <p className="text-sm">{message.message}</p>
                      </div>
                    </div>
                  ))}
                </div>
                
                <div className="flex gap-2">
                  <Textarea
                    placeholder={`Ask about ${pet.name}'s behavior, upload files, or get care advice...`}
                    value={chatMessage}
                    onChange={(e) => setChatMessage(e.target.value)}
                    className="min-h-[60px]"
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' && !e.shiftKey) {
                        e.preventDefault();
                        handleChatSubmit();
                      }
                    }}
                  />
                  <Button onClick={handleChatSubmit} className="self-end">
                    <Send className="h-4 w-4" />
                  </Button>
                </div>
                
                <div className="mt-4 grid grid-cols-3 gap-2">
                  <Button variant="outline" size="sm" onClick={() => fileInputRef.current?.click()}>
                    <Upload className="h-3 w-3 mr-1" />
                    Audio
                  </Button>
                  <Button variant="outline" size="sm" onClick={() => videoInputRef.current?.click()}>
                    <Video className="h-3 w-3 mr-1" />
                    Video
                  </Button>
                  <Button 
                    variant="outline" 
                    size="sm"
                    className={isRecording ? "bg-destructive text-destructive-foreground" : ""}
                    onClick={handleRecordingToggle}
                    disabled={isAnalyzing}
                  >
                    <Mic className="h-3 w-3 mr-1" />
                    {isRecording ? "Stop" : "Record"}
                  </Button>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </main>

      <Footer />
    </div>
  );
};

export default Analysis;