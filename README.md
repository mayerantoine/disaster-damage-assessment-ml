# Disaster damage assessment app

Damage assessments  during and after natural disasters are core activities for humanitarian organizations. They are used to increase situational awareness during the disaster or to provide actionable information for rescue teams.Post-disasters, they are essential  to evaluate the financial impact and the recovery cost.

However, effectively implementing damage assessment using surveys represents a challenge for several organizations as it takes weeks and months and is logistically difficult for rapid assessment. In recent years a lot of research has been done on using AI and machine learning to automate classification of the damages. Those studies and systems use social media imagery data such twitter text and images, aerial and satellite images to assess and map damages during and after a disaster. For the AWS disaster response hackathon one of the challenge is to answer the question :  HOW MIGHT WE ACCURATELY AND EFFICIENTLY DETERMINE THE EXTENT OF DAMAGE TO INDIVIDUAL HOMES IN A GIVEN DISASTER-IMPACTED AREA ?

To answer that question our team proposed to **build and deploy an edge-based computer vision solution on smartphones for damage assessment**. We plan to  build a simple app as a proof of concept that can :
* Allow users to take a picture of a building, road or bridge  and automatically perform a series of classification :
  * Is it Relevant or not relevant to natural disasters ?
  * Type of natural disaster : Earthquake , Flood and Hurricane 
  * Is it a building Building , Road, Bridge 
  * Severe, mild or no damage (or : no damage,affected, minor,major, destroyed)
* Have a centralized storage for all relevant pictures with predicted classes, that can be used for human review, re-training of the model and central reporting for situational awareness.


We will use Amazon SageMaker to build, train, debug , deploy and export the model for mobile devices.

This project is inspired and built on top of existing work from the manuscripts: “*Damage Assessment from Social Media Imagery Data During Disasters*”  and “*Automatic Image Filtering on Social Networks Using Deep Learning and Perceptual Hashing During Crises*” from ( D. T. Nguyen et al) .
