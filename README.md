# SAR-spaceapps
Using SAR to visualize earth processes created as a part of NASA space apps challenge 2025.

RadarVision: Web-Based SAR Hub for Earth’s Dynamic Processes
Summary
The Integrated SAR Earth Observatory Platform, developed for the NASA Space Apps Challenge 2025, is a web-based dashboard that unifies Synthetic Aperture Radar (SAR) applications into a single, accessible platform to simplify Earth monitoring for non-experts. Built with Python, Streamlit, and Plotly, it integrates real-time data from NASA EONET, USGS, and CelesTrak to track disasters, ground deformation, environmental changes, and SAR satellite missions. Its five tabs—Disaster Monitoring, Ground Deformation, Earth's Heartbeat, Integrated Dashboard, and Mission Planner—offer interactive maps, 3D subsidence models, and real-time analytics, showcasing SAR’s all-weather, high-precision capabilities. By combining these functions, the platform demystifies SAR for diverse users, enabling intuitive insights into floods, earthquakes, subsidence, and environmental trends without requiring technical expertise. This addresses the challenge by making SAR’s complex data accessible, fostering informed decision-making for disaster response and sustainability. Its importance lies in democratizing advanced radar technology, promoting STEM engagement, and supporting global efforts to monitor Earth’s dynamic processes, all through a unified, user-friendly interface.
Project Demonstration
https://drive.google.com/file/d/136K5jIDFqwnLRdoZGc--eWN4NTU9HzqH/view?usp=sharing
Project
https://trisolaris-sar-spaceapp.streamlit.app/
Project Details
The Integrated SAR Earth Observatory Platform visually and creatively demonstrates SAR's multi-purpose capabilities ,disaster monitoring, ground deformation, environmental tracking, and mission planning—through interactive maps, 3D models, and real-time analytics, all unified on a single, intuitive dashboard.

What does it do?

The platform aggregates and visualizes real-time SAR-derived data to track natural disasters, ground deformation, environmental changes, and satellite missions. It fetches live data from public APIs like NASA EONET for disasters (e.g., wildfires, floods), USGS for earthquakes, and CelesTrak for satellite orbital elements. Users interact via a Streamlit interface with five tabs:

Disaster Monitoring: Displays global event maps, change detection imagery (before/after SAR simulations), and recent event lists.
Ground Deformation: Analyzes subsidence for iconic buildings using InSAR techniques, generating time series, horizontal/vertical displacement charts, and 3D subsidence models.
Earth's Heartbeat: Tracks vital signs like ice sheet dynamics or ocean height with time series, trend lines, and frequency spectrum analysis.
Integrated Dashboard: Provides aggregated metrics, timelines, and distributions for quick overviews.
Mission Planner: Calculates SAR satellite passes (e.g., Sentinel-1, ALOS-2) over user-selected locations for acquisition planning. It works by processing API data into interactive Plotly visualizations, with fallbacks to simulated datasets for reliability, ensuring all-weather, day-night insights into Earth's surface.
upload_image.15:32:34.108179
Picture left is a screen grab of the user interface created.

What Benefits Does It Have?
It offers real-time, accessible monitoring without requiring API keys or paid services, enabling users like researchers, emergency responders, or educators to gain insights into disasters and environmental shifts. Benefits include millimeter-precision deformation detection , 24/7 coverage penetrating clouds/darkness, and intuitive visualizations that simplify complex data. It's cost-free, scalable, and educational, promoting broader understanding of SAR's role in Earth observation.

What Is the Intended Impact of the Project?
The platform aims to democratize SAR data access, raising awareness of Earth's vulnerabilities to disasters, subsidence, and climate change. By integrating multi-source data into one tool, it supports informed decision-making for disaster response, urban planning, and environmental policy, potentially reducing risks in vulnerable areas (e.g., coastal cities facing subsidence). Long-term, it inspires STEM engagement and contributes to global sustainability goals, like those aligned with NASA's Earth observation missions.

What Tools, Coding Languages, Hardware, or Software Did You Use to Develop Your Project?
Developed primarily in Python (version 3.9+), the core framework is Streamlit for the web interface. 

Key libraries include NumPy and Pandas for data processing, Plotly for interactive charts/maps, Requests for API calls, and Skyfield for satellite orbital calculations. 

Data sources are free public APIs (NASA EONET, USGS, CelesTrak).

How Is the Project Creative?
The platform creatively unifies SAR applications into a single "observatory," blending real-time data with imaginative visualizations like 3D subsidence "bowls," SAR change imagery, and an "Earth’s Heartbeat" metaphor for environmental trends. The Mission Planner gamifies satellite tracking, while the dashboard’s cohesive design transforms technical data into an engaging narrative, making SAR’s multi-purpose potential visually compelling.


What Factors Were Considered?
We prioritized a unified database for seamless data integration. Ethical considerations ensured public data usage and accurate visualizations to avoid misinformation. 

Technical factors included library compatibility, real-time performance, and scalability.

 We balanced creativity with scientific rigor, grounding simulations in real-world data (e.g., published subsidence rates) to align with NASA’s challenge goals
