var welcome_prompt = "你好，这次购物怎样？";
var welcome_img_addr = "images/welcome.png";
var empty_prompt = "还什么都没有告诉我呢...";
var empty_img_addr = "images/empty.png";
var rare_prompt = "我好像不太懂这些词的意思......";
var rare_img_addr = "images/rare.png";
var good_prompt = "那么这次还算不错？我们会持续改进的！";
var good_img_addr = "images/good.png";
var bad_prompt = "非常抱歉，下次不会再有这样的事情发生了。";
var bad_img_addr = "images/bad.png";
var split_url = "/split_words/";
var predict_url = "/predict/";


function sys_say(text){
	var sys_mes = $("#sys_mes");
	sys_mes.val(sys_mes.val() + text + "\n");
	sys_mes.scrollTop(sys_mes.prop("scrollHeight") - sys_mes.height());
}
function user_say(text){
	sys_say(">> " + $("#sys_mes").val() + "“" + text + "”" + "\n");
}
function pic_change(img_path){
	$("#sys_photo").attr("src", img_path);
}

function show_result(logits){
	if(logits[0] > logits[1]){
		pic_change(bad_img_addr);
		sys_say(bad_prompt);
	}else{
		pic_change(good_img_addr);
		sys_say(good_prompt);
	}
}

function handle_predict(data){
	words = data.data.words;
	logits = data.data.logits;
	rare_words = new Set();
	words_prompt = "";
	for (var word in words){
		if (word[2] == null || word[2].substr(9) == 'RAREWORD_'){
			if (!rare_words.has(word[0])){
				words_prompt += word[0] + ",";
				rare_words.add(word[0]);
			}
		}
	}
	rare_words = rare_words.values();
	if(rare_words.length == 0){
		show_result(logits);
	}else{
		sys_say(rare_prompt);
		pic_change(rare_img_addr);
		sys.say(words_prompt);
		setTimeout("show_result(logits)", 500);
	}
	
}

function handle_split(data){
	data = data.data.words;
	prompt = "";
	for (var word in words){
		prompt += word[0] + ','
	}
	user_say(prompt);
	$.post(split_url, 
		{"words": data}, handle_predict
	);
}

function submit_d(){
	data = $("#user_mes").val();
	if (data.trim() == ""){
		sys_say(empty_prompt);
		pic_change(empty_img_addr);
	}else{
		user_say(data);
		$("#user_mes").val("")
		$.post(split_url, 
			{"text": data}, handle_split
		);
	}
}

$($("#sys_mes").val(welcome_prompt + "\n"));
$($("#user_submit").click(submit_d));

