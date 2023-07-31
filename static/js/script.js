const options = {
  method: "GET",
  headers: {
    "X-RapidAPI-Key": "4d1ff3bbc4mshc30cd20261f3a1cp14a3f8jsn827a52108d5a",
    "X-RapidAPI-Host": "weather-by-api-ninjas.p.rapidapi.com",
  },
};

const getWeather = (city) => {
  cityname.innerHTML = city;
  fetch(
    "https://weather-by-api-ninjas.p.rapidapi.com/v1/weather?city=" + city,
    options
  )
    .then((response) => response.json())
    .then((response) => {
      console.log(response);

      temp2.innerHTML = response.temp;
      temp.innerHTML = response.temp;
      feels_like.innerHTML = response.feels_like;
      humidity2.innerHTML = response.humidity;
      humidity.innerHTML = response.humidity;
      min_temp.innerHTML = response.min_temp;
      max_temp.innerHTML = response.max_temp;
      wind_speed2.innerHTML = response.wind_speed;
      wind_speed.innerHTML = response.wind_speed;
      wind_degrees.innerHTML = response.wind_degrees;
      sunrise.innerHTML = response.sunrise;
      sunset.innerHTML = response.sunset;
    })
    .catch((err) => console.error(err));
};
submit.addEventListener("click", (e) => {
  e.preventDefault();
  getWeather(city.value);
});
getWeather("vellore");
