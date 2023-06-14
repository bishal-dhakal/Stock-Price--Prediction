import { useEffect, useState } from 'react';


function Predictions() {

  const [price,setPrice] = useState([]);

  const baseURL ="http://localhost:5000/"

  useEffect(()=> {
    const fetchData = async () =>{
      await fetch(`${baseURL}`)
      .then(res =>res.json())
      .then(data =>{
        // console.log(data)
        setPrice({price:data})
      }).catch((error) => {
          console.log(error);
        });
    };
    fetchData()
  }, [])

  console.log(Object.values(price))

  return (
    <div className="Predictions">
      {price && Object.keys(price).map((value,index)=>{
       return (
       <li key={index}></li>
     );})}
    </div>
  );
}

export default Predictions;
