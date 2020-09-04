import React, { Fragment, useState } from 'react';
import Message from './Message';
import Progress from './Progress';
import axios from 'axios';
import Form from 'react-bootstrap/Form';
import Col from 'react-bootstrap/Col';
import Container from 'react-bootstrap/Container';
import Row from 'react-bootstrap/Row';
import Button from 'react-bootstrap/Button';
import 'bootstrap/dist/css/bootstrap.css';
import Title from './Title'
import schema from './schema';
import Table from './Table';
import './App.css'
import BootstrapTable from 'react-bootstrap-table-next';
import paginationFactory from 'react-bootstrap-table2-paginator';

const FileUpload = () => {
  const [file, setFile] = useState('');
  const [filename, setFilename] = useState('Choose File');
  const [uploadedFile, setUploadedFile] = useState({});
  const [message, setMessage] = useState('');
  const [uploadPercentage, setUploadPercentage] = useState(0);
  const [result, setResult] = useState([]);
  const [isLoading, setIsLoading] = useState(false)

  const onChange = e => {
    setFile(e.target.files[0]);
    setFilename(e.target.files[0].name);
  };


  const onSubmit = async e => {
    e.preventDefault();
    const formData = new FormData();
    formData.append('file', file);
    setIsLoading(true);
      try {
          const res = await axios.post('/api/upload', formData, {
              headers: {
                  'Content-Type': 'multipart/form-data'
              },
              onUploadProgress: progressEvent => {
                  setUploadPercentage(
                      parseInt(
                          Math.round((progressEvent.loaded * 100) / progressEvent.total)
                      )
                  );

                  // Clear percentage
                  setTimeout(() => setUploadPercentage(0), 10000);
              }
          });

          // const {fileName, filePath} = res.data;
          setResult(res.data);
          console.log(res.data);

          // setUploadedFile({fileName, filePath});

          setMessage('File Uploaded');
          setIsLoading(false);

      } catch (err) {
          if (err.response.status === 500) {
              setMessage('There was a problem with the server');
          } else {
              setMessage(err.response.data.msg);
          }
      }

  };

  const onDownload = e => {
      e.preventDefault();

	var json = JSON.stringify(result);
    var blob = new Blob([json], {type: "application/json"});
    var url  = URL.createObjectURL(blob);
	let a = document.createElement('a');
	a.href = url;
	a.download = 'results.json';
	a.click();
  };

  // const Test = ({result}) => (
  // <>
  //   {result.map(res => (
  //     <div className="results" key={res.Modulation}>{res.Modulation}</div>
  //   ))}
  // </>
  //   );

    const columns= [
    {
        dataField: 'Modulation',
        text: 'Modulation'
    }];

    const options = {
                        page: 2,
                        sizePerPageList: [ {
                          text: '5', value: 5
                        }, {
                          text: '10', value: 10
                        }, {
                          text: 'All', value: result.length
                        } ],
                        sizePerPage: 5,
                        pageStartIndex: 0,
                        paginationSize: 3,
                        prePage: 'Prev',
                        nextPage: 'Next',
                        firstPage: 'First',
                        lastPage: 'Last',
                      };


  return (
    <Fragment>
      <Container>
           <div><h1><Title /></h1></div>

         <div className="container">
         <div className="row">

           <div className="col-md-6 center">
             <Form method="post">

                 <Row className={"row-elements"}>
                    <Col style={{paddingRight: '50px', paddingTop:'-30px'}}>
               <div className="form-group files color">

                 <label className={"upload-label"} style={{color: "white"}}>
                     Upload Your .npz File
                 </label>
                 <input type="file" className="form-control"
                 onChange={onChange}/>

                 <Progress percentage={uploadPercentage} />

                   <button type="button" className="btn btn-success btn-block"
                           disabled={isLoading}
                          onClick={!isLoading ? onSubmit : null}>
                          { isLoading ? 'Predicting' : 'Upload' }

                   </button>

                   {/*<button type="button" className="btn btn-success btn-block" onClick={onSubmit}>*/}
                   {/*    Predict Modulation*/}
                   {/*</button>*/}

                   {message ? <Message msg={message} /> : null}


               </div>
                    </Col>

                     <Col style={{paddingLeft: '50px', paddingTop: '0px'}}>

                  <div  className={"table-div"} style={{ marginTop: 0, color: 'white'}} >
                <BootstrapTable
                    headerStyle= {{ backgroundColor: 'white'}}
                    rowStyle={ { color: 'white' } }
                    striped
                    hover
                    keyField='id'
                    data={ result }
                    columns={ columns }
                    pagination={ paginationFactory(options)}
                />
                    </div>

                         <label style={{color: "white"}}>
                             Download results: </label>
				<Button variant="primary" onClick={onDownload}>Download</Button>
             </Col>
                 </Row>

             </Form>


           </div>
         </div>
       </div>
       </Container>

    </Fragment>
  );
};

export default FileUpload;