## Inspiration

We were inspired by our own lives. Hunched over hours of schoolwork, we noticed our posture slowly worsening and our eyes becoming strained from long, late night study sessions. Given the opportunity to focus on the home life and bring some improvement, we decided to develop an application that would benefit students just like us. Additionally, we wanted to help people like our parents, facing similar issues hunched over their work from home.

## What it does

AL-I (Analytics Lifestyle Interface), which is a pun on the word Ally, as we hope this application will be for our users, utilizes video detection software to obtain certain landmarks on the user's body. For example, our application detects the position of your eyelid, ear, shoulder, hip, etc. Then, based on the position and angles between the different landmarks, the application is capable of identifying when you have a bad posture, are straining your eyes, etc. and for how long. Then, if the app notices trends, it pushes a notification to the user's device asking the 

We used MediaPipe and openCV to extract the landmarks from the user's camera and to overlay the relative positions and angles on top of the video itself to enable layman analyzation. Additionally, the video system also analyzes the user's environment (such as their lighting) to see if they are in a location which can increase their body's stress, strain, etc. and recommends suggestions.

For example, one of our datapoints is the user's EAR value, which determines how open their eyes are to be used for fatigue. Here is the equation we had to use:

$$\\text{EAR} = \frac{||p_2 - p_6| + |p_3 - p_5||}{2|p_1 - p_4|}$$

Additionally, to find the proximity of the shoulders to the ear (in order to detect sustained shoulder elevation), we used this equation:

$$\text{Ratio} = \left( \frac{|\text{Shoulder}_y - \text{Ear}_y|}{\text{Torso Height}} \right) \times 100$$

Furthermore, in order to detect the overall brightness of the scene, we used a series of the different pixel’s intensities, which can be represented by this:

$$\text{Brightness} = \frac{1}{n} \sum_{i=1}^{n} P_i$$

Additionally, in order to obtain the angle of inclination for their cervical (head) posture, we utilized this equation:

$$\theta = \tan^{-1} \left( \frac{\Delta y}{\Delta x} \right)$$

Lastly, we connected a Gemini API key to the application which analyzes the collected data and recognizes trends and incorporates more long-term changes the user can make.

## Challenges we ran into

While we originally thought splitting into different teams would be a good way to ensure work runs as smoothly, we faced issues when we had to merge all of our code. Some of us worked on the backend, with little care for the front end, while others worked on the front end, with little care for the backend making this process very tedious. There were times we thought that we wouldn't finish with the stress that we were facing. All of our concerns were alleviated when one of our laptops suddenly got a notification telling us to sit up straight; our app had worked!
