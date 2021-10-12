const button = document.querySelector('button')
const input = document.querySelector('input')
const div = document.querySelector('div')

const arr = [1,2,3,4,5]
const val= JSON.stringify(arr)
console.log(arr)
console.log(val)

localStorage.setItem("val", JSON.stringify(arr))
console.log(JSON.parse(localStorage.getItem("val")))