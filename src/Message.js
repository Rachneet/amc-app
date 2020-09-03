import React, {useState} from 'react';
import PropTypes from 'prop-types';
import './App.css';
import 'bootstrap/dist/css/bootstrap.css';
import Button from 'react-bootstrap/Button';
import Alert from 'react-bootstrap/Alert';


const Message = ({ msg }) => {
  const [show, setShow] = useState(true);

  if (show) {
    return (
      <Alert variant="warning" onClose={() => setShow(false)} dismissible>
          {msg}
      </Alert>
    );
  }
  return <Button className={"alert-button"} onClick={() => setShow(true)}>Show Alert</Button>;
}


Message.propTypes = {
  msg: PropTypes.string.isRequired
};

export default Message;