FaceFolders = {'TestFaces','Angry', 'Contempt', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise'};

%Predetermined face crop size
CropSize = 100;

%Prepare Total SumFace to calculate Total MeanFace
TotalEmotionSumFace = zeros(CropSize, CropSize);
TotalNumFaces = 0; %counter for total number of faces 

for j = 1:length(FaceFolders)
    
    %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    %Select an Emotion folder from the folder list (Angry, Contempt,
    %Disgust, Fear, Happy, Sad, Surprise)
    Emotion = FaceFolders{j};
    
    %Initialize array to hold uncropped Happy Faces
    FacesRaw = zeros(1, 490, 640);
    
    %read in all images
    TrainingImages = Emotion;
    Training_Folder = dir(fullfile(TrainingImages, '*.png'));
    for j=1:length(Training_Folder)
        ImCrop = im2double(imread(fullfile(TrainingImages, Training_Folder(j).name)));  % Read image
        FacesRaw(end+1,:,:) = ImCrop;
        imshow(ImCrop)
        title(Emotion);
    end
    
    FacesRaw = FacesRaw(2:end,:,:);
    
    %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    
    %Initialize array to hold cropped Happy Faces
    CroppedFaces = zeros(1, CropSize, CropSize);
    
    %Detect faces using Viola-Jones (Haar features) detector and crop them to
    %predetermined size
    
    FacesSize = size(FacesRaw,1);
    
    TotalNumFaces = TotalNumFaces + FacesSize;
    
    %Object detector for front face using CART based classifier
    faceDetector = vision.CascadeObjectDetector('FrontalFaceCART', 'MergeThreshold', 5);
    
    
    for i = 1:FacesSize
        
        face = FacesRaw(i,:,:);
        face = squeeze(face);
        
        %find face in each image
        faceBox = step(faceDetector, face);
        IFaces = insertObjectAnnotation(face, 'rectangle', faceBox, 'Face');
        imshow(IFaces), title(strcat(Emotion,' Detected face'));
        pause(1)
        
        row = faceBox(2);
        col = faceBox(1);
        boxsize = faceBox(3);
    end
    
end

