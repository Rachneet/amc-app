(this["webpackJsonpamc-app"]=this["webpackJsonpamc-app"]||[]).push([[0],{139:function(e,t,a){},14:function(e,t,a){},141:function(e,t,a){"use strict";a.r(t);var n=a(0),r=a.n(n),l=a(12),c=a.n(l),o=a(33),s=a.n(o),i=a(58),u=a(11),m=(a(14),a(35),a(22)),p=a(64),d=function(e){var t=e.msg,a=Object(n.useState)(!0),l=Object(u.a)(a,2),c=l[0],o=l[1];return c?r.a.createElement(p.a,{variant:"warning",onClose:function(){return o(!1)},dismissible:!0},t):r.a.createElement(m.a,{className:"alert-button",onClick:function(){return o(!0)}},"Show Alert")},b=function(e){var t=e.percentage;return r.a.createElement("div",{className:"progress"},r.a.createElement("div",{className:"progress-bar progress-bar-striped bg-success",role:"progressbar",style:{width:"".concat(t,"%")}},t,"%"))},g=a(59),f=a.n(g),E=a(65),v=a(19),h=a(62),w=a(63);var j=function(){return r.a.createElement("div",{className:"Title"},r.a.createElement("h1",null,"Signal Modulation Predictor"))},O=(a(92),a(60)),y=a.n(O),N=a(61),x=a.n(N),S=function(){var e=Object(n.useState)(""),t=Object(u.a)(e,2),a=t[0],l=t[1],c=Object(n.useState)("Choose File"),o=Object(u.a)(c,2),p=(o[0],o[1]),g=Object(n.useState)({}),O=Object(u.a)(g,2),N=(O[0],O[1],Object(n.useState)("")),S=Object(u.a)(N,2),P=S[0],k=S[1],C=Object(n.useState)(0),F=Object(u.a)(C,2),T=F[0],U=F[1],D=Object(n.useState)([]),L=Object(u.a)(D,2),z=L[0],J=L[1],M=Object(n.useState)(!1),I=Object(u.a)(M,2),R=I[0],A=I[1],B=function(){var e=Object(i.a)(s.a.mark((function e(t){var n,r;return s.a.wrap((function(e){for(;;)switch(e.prev=e.next){case 0:return t.preventDefault(),(n=new FormData).append("file",a),A(!0),e.prev=4,e.next=7,f.a.post("/api/upload",n,{headers:{"Content-Type":"multipart/form-data"},onUploadProgress:function(e){U(parseInt(Math.round(100*e.loaded/e.total))),setTimeout((function(){return U(0)}),1e4)}});case 7:r=e.sent,J(r.data),console.log(r.data),k("File Uploaded"),A(!1),e.next=17;break;case 14:e.prev=14,e.t0=e.catch(4),500===e.t0.response.status?k("There was a problem with the server"):k(e.t0.response.data.msg);case 17:case"end":return e.stop()}}),e,null,[[4,14]])})));return function(t){return e.apply(this,arguments)}}(),Y={page:2,sizePerPageList:[{text:"5",value:5},{text:"10",value:10},{text:"All",value:z.length}],sizePerPage:5,pageStartIndex:0,paginationSize:3,prePage:"Prev",nextPage:"Next",firstPage:"First",lastPage:"Last"};return r.a.createElement(n.Fragment,null,r.a.createElement(h.a,null,r.a.createElement("div",null,r.a.createElement("h1",null,r.a.createElement(j,null))),r.a.createElement("div",{className:"container"},r.a.createElement("div",{className:"row"},r.a.createElement("div",{className:"col-md-6 center"},r.a.createElement(E.a,{method:"post"},r.a.createElement(w.a,{className:"row-elements"},r.a.createElement(v.a,{style:{paddingRight:"50px",paddingTop:"-30px"}},r.a.createElement("div",{className:"form-group files color"},r.a.createElement("label",{className:"upload-label",style:{color:"white"}},"Upload Your .npz File"),r.a.createElement("input",{type:"file",className:"form-control",onChange:function(e){l(e.target.files[0]),p(e.target.files[0].name)}}),r.a.createElement(b,{percentage:T}),r.a.createElement("button",{type:"button",className:"btn btn-success btn-block",disabled:R,onClick:R?null:B},R?"Predicting":"Upload"),P?r.a.createElement(d,{msg:P}):null)),r.a.createElement(v.a,{style:{paddingLeft:"50px",paddingTop:"0px"}},r.a.createElement("div",{className:"table-div",style:{marginTop:0,color:"white"}},r.a.createElement(y.a,{headerStyle:{backgroundColor:"white"},rowStyle:{color:"white"},striped:!0,hover:!0,keyField:"id",data:z,columns:[{dataField:"Modulation",text:"Modulation"}],pagination:x()(Y)})),r.a.createElement("label",{style:{color:"white"}},"Download results: "),r.a.createElement(m.a,{variant:"primary",onClick:function(e){e.preventDefault();var t=JSON.stringify(z),a=new Blob([t],{type:"application/json"}),n=URL.createObjectURL(a),r=document.createElement("a");r.href=n,r.download="results.json",r.click()}},"Download")))))))))},P=function(){return r.a.createElement("div",{className:"container mt-4"},r.a.createElement(S,null))};a(139);c.a.render(r.a.createElement(P,null),document.getElementById("root"))},67:function(e,t,a){e.exports=a(141)},92:function(e){e.exports=JSON.parse("{}")}},[[67,1,2]]]);
//# sourceMappingURL=main.dcea8148.chunk.js.map