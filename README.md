<h1>face-mask detection and Warning System</h1>
<h3>Facemask detection program and pre-trained weights!</h3>
<p> This is my first project on the live object detection, so please feel free to suggest changes where necessary </p>

<b>The code was made for the Colab environment so when running offline make appropriate changes.</b>
<ul>
  <li>Project inspired from : <a href= https://github.com/aieml >perceptron </a> </li>
  <li>Dataset from :<a href =https://github.com/ohyicong/masksdetection/tree/master/dataset/without_mask> Ohyicong</a></li>
  <li> For full augmented dataset visit <a href = 'https://drive.google.com/drive/folders/1hIu0WsaiZFrZaenOoQlY-i2AXJgzhPt2?usp=sharing'>my drive</a>
 </ul>

<i>Note : The model uses a CNN architecture so errors due to outside noise may cause variation from expected output.</i>

<b> Edit : </b> As part of a hackathon we decided to modify the program a little further to include face recogninition and messaging functions. This project is now a joined effort of <a href= "https://github.com/Vysakh-T">Vyshak T</a> ,Azeem M Basheer , Kallu Sudarshan and Martin Joe C.

<h3>Program Structure</h3>
<h4><ol>
  <li> Load the pretrained model </li>
  <li> Find the ROI with face using haarcascade </li>
  <li> Detect if mask present or not</li>
  <li> If mask present display "Mask is present".</li>
  <li> If mask not present , check for the persons identity using facial recognition library</li>
  <li> Using messaging script, send a whatsapp warning </li>
  <li> display the video footage with bounting box and labels around detected faces</li>
  </ol></h4>
    
