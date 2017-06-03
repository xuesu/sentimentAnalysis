var welcome_prompt = "你好，这次购物怎样？";
var welcome_img_addr = "/static/images/welcome.png";
var empty_prompt = "还什么都没有告诉我呢...";
var empty_img_addr = "/static/images/empty.png";
var rare_prompt = "我好像不太懂这些词的意思......";
var rare_img_addr = "/static/images/rare.png";
var prompts = {
    "-1": "非常抱歉，下次不会再有这样的事情发生了。",
    "0": "那么这次还算不错？我们会持续改进的！",
    "1": "谢谢光临，争取下次给您留下更好的印象"
};
var img_addrs = {
    "-1":"/static/images/bad.png",
    "0":"/static/images/good.png",
    "1":"/static/images/bad.png"
}
var predict_url = "/predict/";


function sys_say(text){
	var sys_mes = $("#sys_mes");
	sys_mes.val(sys_mes.val() + text + "\n");
	sys_mes.scrollTop(sys_mes.prop("scrollHeight") - sys_mes.height());
}
function user_say(text){
	sys_say(">> “" + text + "”" + "\n");
}
function pic_change(img_path){
	$("#sys_photo").attr("src", img_path);
}

function show_result(tag){
	pic_change(img_addrs[tag.toString()]);
	sys_say(prompts[tag.toString()]);
	$("#loading").fadeOut(500);
}

function handle_predict(data){
	words = data.data.words;
	tag = data.data.tag;
	var rare_words = []
	for(var word in words){
	    for(var info in word){
	        if(info == "RareWord"){
	            rare_words.push(word[0]);
	            break;
	        }
	    }
	}
	if(rare_words.length == 0){
		show_result(tag);
	}else{
		sys_say(rare_prompt);
		pic_change(rare_img_addr);
		sys_say(rare_words);
		setTimeout("show_result(logits)", 500);
	}

}

model_type = "NOLSTM";
col_name = "ctrip";

function submit_d(){
	data = $("#user_mes").val();
	if (data.trim() == ""){
		sys_say(empty_prompt);
		pic_change(empty_img_addr);
	}else{
		user_say(data);
		$("#loading").fadeIn(500);
		$("#user_mes").val("")
		$.post(predict_url,
			{"text": data, "model_type": model_type, "col_name": col_name},
			handle_predict
		);
	}
}

$($("#sys_mes").val(welcome_prompt + "\n"));
$($("#user_submit").click(submit_d));
$($("#sec_en").click(function(){
    col_name = "nlpcc_en";
}));
$($("#sec_zh").click(function(){
    col_name = "ctrip";
}));
$($("#sec_cnnpl").click(function(){
    model_type = "NOLSTM";
}));
$($("#sec_cnnlstmpl").click(function(){
    model_type = "LSTM";
}));