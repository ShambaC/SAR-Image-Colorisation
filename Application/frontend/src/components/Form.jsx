import React from "react";
import { useState, useRef } from "react";

const Form = () => {
  const [selectedImage, setSelectedImage] = useState(null);
  const [previewURL, setPreviewURL] = useState(null);
  const [lat, setLat] = useState("");
  const [long, setLong] = useState("");
  const [imgDate, setImgDate] = useState("");
  const [isGenerating, setIsGenerating] = useState(false);
  const [generatedImageURL, setGeneratedImageURL] = useState(null);


  const inputFileRef = useRef();

  const handleImageChange = (e) => {
    const file = e.target.files[0];
    const fileType = file.type;
    
    const validTypes = ["image/jpeg", "image/jpg", "image/png"];

    if (validTypes.includes(fileType)) {
      setSelectedImage(file);
      setPreviewURL(URL.createObjectURL(file));
    } else {
      alert("Only .jpeg, .jpg and .png file formats are allowed.");
    }
  };

  const handleGenerate = async(event) =>{
    event.preventDefault();
    if(!selectedImage || !lat || !long || !imgDate){
      alert("Please fill out all the required fields.")
      return;
    }

    const data = new FormData();
    data.append('image', selectedImage);
    data.append('lat', lat);
    data.append('long', long);
    data.append('imgDate', imgDate);

    try{
      setIsGenerating(true);
      const response = await fetch("http://localhost:5000/generate", {
        method: "POST",
        body: data
      });
      const result = await response.json();
      alert(`Status: ${result.status}\nMessage: ${result.message}`);
      setSelectedImage(null);
      setPreviewURL(null);
      setLat("");
      setLong("");
      setImgDate("");
      if (inputFileRef.current) inputFileRef.current.value = null;
      setGeneratedImageURL(`http://localhost:5000/image/${result.imageUrl}`);
    } catch (error) {
      console.error("Error during generation:", error);
      alert("Something went wrong!");
    } finally{
      setIsGenerating(false);
    }
  }

  return (
      <form action="POST" id="inputForm">
        <div className="imageSection">
          <label htmlFor="uploadImage">Upload Image: </label>
          <input
            type="file"
            name="inpImage"
            id="inpImage"
            accept=".jpg, .jpeg, .png"
            onChange={handleImageChange}
            ref={inputFileRef}
            required
          />
          {previewURL && (
            <>
            <div className="previewImage">
              <h2>Preview</h2>
              <img
                src={previewURL}
                alt="Uploaded Image"
              />
            </div>
            <div className="clearImage">
              <button onClick={() => {
                setSelectedImage(null);
                setPreviewURL(null);
                if(inputFileRef.current){
                  inputFileRef.current.value = null;
                }
              }}>
              Clear  
              </button>
            </div>
            </>
          )}
        </div>
        <div className="imageInfo">
          <div className="coords">
            <div className="lat">
              <label htmlFor="lat">Latitude: </label>
              <input type="number" name="lat" id="lat" value={lat} required onChange={(e) => setLat(e.target.value)}/>
            </div>
            <div className="long">
              <label htmlFor="long">Longitude: </label>
              <input type="number" name="long" id="long" value={long} required onChange={(e) => setLong(e.target.value)}/>
            </div>
          </div>
          <div className="date">
            <label htmlFor="imgDate">Date: </label>
            <input type="date" name="imgDate" id="imgDate" value={imgDate} required onChange={(e) => setImgDate(e.target.value)}/>
          </div>
        </div>
        <div className="generate">
          <button onClick={handleGenerate} disabled={isGenerating}>{isGenerating ? "Generating..." : "Generate"}</button>
        </div>
        {generatedImageURL && (
          <div className="generatedPreview">
            <h3>Generated Image Preview</h3>
            <img src={generatedImageURL} alt="Generated output" style={{ maxWidth: "256px", marginTop: "1rem" }} />
          </div>
        )}
      </form>
  );
};

export default Form;
