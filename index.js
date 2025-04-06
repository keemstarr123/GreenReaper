/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
// [START maps_drawing_tools]
// This example requires the Drawing library. Include the libraries=drawing
// parameter when you first load the API. For example:
// <script src="https://maps.googleapis.com/maps/api/js?key=YOUR_API_KEY&libraries=drawing">
async function initMap(sessiondata = null) {
    let lastCircle = null;
    let hidden_input = document.getElementById("location");
    const [{ Map }, { AdvancedMarkerElement,  Geocoder }] = await Promise.all([
      google.maps.importLibrary("marker"),
      google.maps.importLibrary("places"),
      google.maps.importLibrary("geocoding")
    ]);
    const geocoder = new google.maps.Geocoder();;
    if (document.getElementById("map")) {
      const mapelement = document.getElementById("map");

      const map = new google.maps.Map(mapelement, {
        center: { lat: 3.1319, lng: 101.6841 },
        zoom: 16,
        mapTypeId: 'hybrid',
        mapTypeControl: false, 
        streetViewControl: false,
        fullscreenControl: false 
      });

      
      const drawingManager = new google.maps.drawing.DrawingManager({
        drawingMode: google.maps.drawing.OverlayType.MARKER,
        drawingControl: true,
        drawingControlOptions: {
          position: google.maps.ControlPosition.TOP_CENTER,
          drawingModes: [
            google.maps.drawing.OverlayType.CIRCLE,
          ],
        },
        markerOptions: {
          icon: "https://developers.google.com/maps/documentation/javascript/examples/full/images/beachflag.png",
        },
        circleOptions: {
          fillColor: "#ffff00",
          fillOpacity: 0.2,
          strokeWeight: 3,
          clickable: true,
          editable: true,
          zIndex: 1,
        },
      });
      drawingManager.setMap(map);

      // Add Autocomplete - Traditional Method
      const input = document.createElement('input');
      input.id = 'pac-input';
      input.class = "form-control";
      input.style.height = "7%";
      input.style.width = "40%";
      input.type = "text";
      input.placeholder = "Search for places";
      
      const searchBox = new google.maps.places.SearchBox(input);
      map.controls[google.maps.ControlPosition.TOP_LEFT].push(input);

      searchBox.addListener("places_changed", () => {
          const places = searchBox.getPlaces();
          if (places.length === 0) return;
          
          const place = places[0];
          if (!place.geometry) return;
          
          map.panTo(place.geometry.location);
      });

    
      drawingManager.setMap(map);
      google.maps.event.addListener(drawingManager, 'circlecomplete', (circle) => {
          console.log('Circle complete listener triggered');
          if (lastCircle) {
              lastCircle.setMap(null); // Remove previous circle
          }

          lastCircle = circle;
          
          // Get circle details
          const center = circle.getCenter();
          const radius = circle.getRadius();
          hidden_input.value = `${center.lat()},${center.lng()},${radius}`;
          console.log("Circle drawn:", {
              center: { lat: center.lat(), lng: center.lng() },
              radius: radius,
              area: Math.PI * radius * radius // Calculate area in sq meters
          });



          // Optional: Add click listener to the circle
          circle.addListener('click', () => {
              console.log('Circle clicked', circle.getRadius());
          });
      });
    } else {
      function reverseGeocode(lat, lng, callback) {
        geocoder.geocode({ location: { lat, lng } }, (results, status) => {
            if (status === "OK") {
                if (results[0]) {
                    const address = results[0].formatted_address;
                    const components = results[0].address_components;
                    callback(address, components);
                } else {
                    console.log("No results found");
                    callback(null, null);
                }
            } else {
                console.log("Geocoder failed due to:", status);
                callback(null, null);
            }
        });
    }
    var formdata = JSON.parse(sessionStorage.getItem("tempData"));
    
    // Usage
    reverseGeocode(formdata['lat'],formdata['lng'], (address, components) => {
        if (address) {
            document.getElementById("full_location").textContent = address;
            console.log("Address Components:", components);
        }
    });




      const mapelement = document.getElementById("map2");

      const map = new google.maps.Map(mapelement, {
        center: { lat: formdata['lat'], lng: formdata['lng'] },
        zoom: 16,
        mapTypeId: 'hybrid',
        mapTypeControl: false, 
        streetViewControl: false,
        fullscreenControl: false 
      });

      
      const drawingManager = new google.maps.drawing.DrawingManager({
        drawingMode: google.maps.drawing.OverlayType.MARKER,
        drawingControl: false,
        
        circleOptions: {
          fillColor: "#ffff00",
          fillOpacity: 0.2,
          strokeWeight: 3,
          clickable: true,
          editable: true,
          zIndex: 1,
        },
      });

    
      drawingManager.setMap(map);
      const circle = new google.maps.Circle({
        strokeWeight: 3,
        fillColor: "#FFFF00",
        fillOpacity: 0.2,
        map: map,
        center: { lat: formdata['lat'], lng: formdata['lng'] },
        radius: formdata['radius'] // in meters
      });
      
    }
    
}

  
window.initMap = initMap;
// [END maps_drawing_tools]

document.getElementById('nextbutton').addEventListener('click', () => {
  var budget = document.querySelector('[name="budget"]').value;
  var residential_type = document.querySelector('[name="residential_type"]').value;
  var location = document.querySelector('[name="location"]').value.split(",");
  var container = document.querySelector(".right-con");
  var lng = parseFloat(location[1]);
  var lat = parseFloat(location[0]);
  var radius = parseFloat(location[2]);

  sessionStorage.setItem("tempData", JSON.stringify ({
    budget: budget,
    residential_type: residential_type,
    lat : lat,
    lng : lng,
    radius: radius
  }));

  container.innerHTML = `
  <div class="row g-0 ms-0 ps-0" style="height:6%">
    <div class="col-12 h-100 d-flex justify-content-start">
        <div>
            
                <span class="border border-2 rounded-3 border-dark material-symbols-outlined fs-1">
                    check
                    </span>
        </div>
        <div class="ms-3">
            <div class="col-12 sub-title">
                Additional Information
            </div>
            <div class="col-12 sub-description">
                Ensure that the information entered is correct.
            </div>
        </div>
    </div>
  </div>
  <div class="row g-0 map d-flex" style="height:60%">
    <span class="fw-bold">Location:</span>
    <div id="map2"></div>
    <br>
    <i style="text-align: end;">Location: <span id="full_location"></span></i>
    <input type="hidden" id = "location" name = "location">
  </div>
  <div class="flex-column g-0 d-flex" style="height:15%">
    <div class="fw-bold">Special request:</div>
    <div style="flex: 1; display: flex;">
        <textarea name="" class="form-control" id="" style="flex: 1; resize: none; width: 100%; height: 100%;" placeholder="Malaysian's style.">
        </textarea>
    </div>
  </div>
  <div class="row g-0 mb-0" style="height:6%">
    <div class="col-6 p-2">
        <input class ='btn col-12 btn-outline-secondary opacity-50' type="reset" value="Reset">
    </div>
    <div class="col-6 p-2">
        <input class ='btn col-12 btn-purple' type="Submit" value="Submit">
    </div>
  </div>
  <input type="hidden" name="lat" value = ${lat}>
  <input type="hidden" name="lng" value = ${lng}>
  <input type="hidden" name="residential_type" value = ${residential_type}>
  <input type="hidden" name="budget" value = ${budget}>
  `
  const script = document.createElement('script');
  script.src = `https://maps.googleapis.com/maps/api/js?key=AIzaSyCJArOGKcQAr8ZLPl1LLQOBOI-YGHCZuc8&libraries=drawing,places`;
  script.async = true;
  script.defer = true;
  
  script.onload = () => {
    // Now explicitly call initMap
    initMap().then(() => {
      console.log('Map initialized successfully');
    });
  };

  document.body.appendChild(script);
  
});




