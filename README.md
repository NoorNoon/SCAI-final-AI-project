# SCAI-final-AI-project
Smart Stadium Management System Overview A comprehensive AI-powered system developed to enhance the live sports experience for both fans and organizers in smart stadiums. This solution focuses on real-time emotion analysis, intelligent crowd behavior tracking, personalized interaction, and safety monitoring—using advanced computer vision techniques and YOLOv8 models. The system bridges the physical and digital experience, providing interactive features for fans and data-driven tools for event management.

The platform includes a suite of Python-based modules such as faceRecognition.py for verifying audience identity, fight.py to detect aggressive incidents, and EmptySeats.py to analyze seat occupancy. Movement and flow are monitored via MovementDetection.py and Bestpath.py, while faintConditions.py helps detect potential medical emergencies. Modules like hand.py recognize gestures, people_tracking.py handles live tracking, and parking.py supports parking availability detection. In parallel, fire.py and peoplePositions.py are under refinement to support hazard detection and crowd mapping.

Fans interact with the system via a web app that includes a smart chatbot (stadium-assistant), an emoji-based feedback system, and dynamic UI themes that adapt to user preferences (e.g., swimming = blue, basketball = orange). The system also offers advanced post-event analytics to summarize fan engagement and deliver personalized digital souvenirs such as highlight videos or NFTs based on emotional responses.

All detections are powered by YOLOv8 models (best.pt, yolov8n.pt) and integrated through scripts such as yolo_media_detector.py, yoloWebCam.py, and train_model.py. The system’s modular design ensures scalability and real-time response, making it ideal for deployment in large stadiums and smart city events.

Key Features • Real-Time Emotion Detection: AI-powered emotion recognition using facial expression analysis across the crowd. • Crowd Monitoring & Alerts: Instant detection of congestion, fainting, fights, and unusual behavior. • Gesture Recognition: Detects hand gestures for crowd interaction and emergency signals. • Personalized UI Themes: Interface adapts to user preferences to enhance emotional connection. • Interactive Surveys: Emoji-based post-event surveys to measure fan satisfaction. • Smart Ticketing: Supports digital seat selection and check-in. • Crowd Flow Optimization: Suggests optimal movement paths and monitors real-time flow. • Post-Event Analytics: Summary of fan engagement with emotion-based highlights and reactions. • AI Assistant (Chatbot): Responds to user inquiries inside the app with context-aware assistance. • Digital Souvenirs: Offers emotional-based memories (videos, images, NFTs) after the event.

Fully Functional Modules

Module Name Functionality Description main.py Central entry point for launching modules. faceRecognition.py Facial recognition for identity and engagement. EmptySeats.py Detects if a seat is occupied or not via bounding box analysis. MovementDetection.py Detects crowd movement and motion heatmaps. Bestpath.py Calculates best movement paths inside the venue. Fight.py Detects fights, aggressive behavior, or physical conflicts. faintConditions.py Detects fainting incidents using posture and sudden collapse tracking. hand.py Recognizes fan hand gestures (waving, pointing, etc). people_tracking.py Real-time tracking of individuals in the crowd. parking.py Identifies available parking spots (basic version, in progress). fire.py Under development for visual fire detection. peoplePositions.py Visualizes the exact positions of people in seating layouts. ShortCodeForFire.py Experimental mini-version of the fire detection logic. traffic_data.csv Dataset used for movement and crowd training. train_model.py YOLOv8 training logic for detection models. YOLO Integration • best.pt, yolov8n.pt: Pre-trained YOLOv8 models used across detection tasks. • yolo_media_detector.py: Applies detection to media files (images/videos). • yoloWebCam.py: Applies detection live via webcam or CCTV.

Web App Project Structure bash Copy Edit /app - Next.js web app /auth - Login and user authentication /dashboard - Admin tools and event control /mobile - Mobile-specific UI /api - API routes and endpoints /components - All reusable UI elements /context - App-wide state management /people-counter - Crowd counting and tracking features /stadium-assistant - Chatbot integration module /ui - UI utilities and visuals /lib - Helper functions and tools /layout - Shared layout templates /public - Static content (icons, logos, etc.) /supabase - Supabase DB and auth config How to Use the Project Extract the ZIP folder into your working directory.

Run the following commands:

bash Copy Edit cd path-to-project npm install npm run dev Open the project at http://localhost:3000 on your browser.

Try the App Directly 🟢 Access the Smart Stadium Web App

Emotion Analysis Models We used two models:

An open-source model from ResEmoteNet

A custom in-house model trained on stadium fan behavior.

Note This project is under active development. While core modules are stable, components like fire detection, advanced positioning, and parking analysis are still being improved. Feedback and collaboration are welcome.
