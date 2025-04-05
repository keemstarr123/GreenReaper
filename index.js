/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
// [START maps_drawing_tools]
// This example requires the Drawing library. Include the libraries=drawing
// parameter when you first load the API. For example:
// <script src="https://maps.googleapis.com/maps/api/js?key=YOUR_API_KEY&libraries=drawing">
async function initMap() {
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
          hidden_input.value = `${center.lat()},${center.lng(), radius}`;
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
    
    // Usage
    reverseGeocode(3.1319, 101.6841, (address, components) => {
        if (address) {
            document.getElementById("full_location").textContent = address;
            console.log("Address Components:", components);
        }
    });




      const mapelement = document.getElementById("map2");

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
        strokeOpacity: 0.2,
        strokeWeight: 3,
        fillColor: "#FFFF00",
        fillOpacity: 0.35,
        map: map,
        center: { lat: 3.1319, lng: 101.6841 },
        radius: 50 // in meters
      });
    }
    
  }

  
  window.initMap = initMap;
  // [END maps_drawing_tools]

