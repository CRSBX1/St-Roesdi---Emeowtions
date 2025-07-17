// Extract audio from video file
export const extractAudioFromVideo = async (videoFile: File): Promise<Blob> => {
  return new Promise((resolve, reject) => {
    const video = document.createElement('video');
    const canvas = document.createElement('canvas');
    const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
    
    video.src = URL.createObjectURL(videoFile);
    video.crossOrigin = 'anonymous';
    
    video.addEventListener('loadedmetadata', async () => {
      try {
        // Create audio source from video
        const source = audioContext.createMediaElementSource(video);
        const mediaRecorder = new MediaRecorder(
          (canvas as any).captureStream().getAudioTracks()[0] ? 
          (canvas as any).captureStream() : 
          await navigator.mediaDevices.getUserMedia({ audio: true })
        );
        
        const chunks: Blob[] = [];
        
        mediaRecorder.ondataavailable = (event) => {
          chunks.push(event.data);
        };
        
        mediaRecorder.onstop = () => {
          const audioBlob = new Blob(chunks, { type: 'audio/webm' });
          resolve(audioBlob);
        };
        
        mediaRecorder.start();
        video.play();
        
        video.addEventListener('ended', () => {
          mediaRecorder.stop();
        });
        
      } catch (error) {
        reject(error);
      }
    });
    
    video.addEventListener('error', () => {
      reject(new Error('Failed to load video file'));
    });
  });
};